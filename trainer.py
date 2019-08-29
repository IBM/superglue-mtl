import os
import json
import torch
import torch.nn as nn
from torch import Tensor
import logging
import time
from tqdm import trange
import numpy as np
from tensorboardX import SummaryWriter
from collections import Iterable
from torch.utils.data import SequentialSampler, DataLoader
import evaluation_utils


logger = logging.getLogger(__name__)


def _run_task_prediction(processed_task_samples, model, args, task_id, device, n_gpu, output_prediction=False,
                         epoch_n=-1, tokenizer=None):
    """Run prediction on given data and task, calculate corresponding metrics.

    Args:
        processed_task_samples:
        model:
        args:
        task_id:
        device:
        output_prediction: bool. Whether to write prediction to file.
        epoch_n: int. When output_prediciton is set to True, create folder with the epoch_n.
        tokenizer: When output_prediction is set to True, use to convert word ids into strings.
    """
    model.eval()
    task_name = args.tasks[task_id]
    softmax = nn.Softmax(dim=-1)
    sigmoid = nn.Sigmoid()
    sampler = SequentialSampler(processed_task_samples)
    data_loader = DataLoader(processed_task_samples, sampler=sampler, batch_size=args.batch_size)
    val_loss = []
    if output_prediction:

        if args.do_predict:
            # run over test set
            fname = "test_%s_epoch_%d.jsonl" % (task_name, epoch_n)
        else:
            fname = "prediction_%s_epoch_%d.jsonl" % (task_name, epoch_n)

        fout = open(args.output_dir + "/%s" % fname, "w")
        if task_name in ["COPA"]:
            val_instance = dict()

    if task_name in ["COPA"]:
        val_preds = dict()
        val_labels = dict()
    else:
        val_preds, val_labels = [], []

    for batch in data_loader:
        batch_tensors = tuple(t.to(device) for t in batch[1:5])

        with torch.no_grad():
            input_ids, input_mask, segment_ids, label_ids = batch_tensors
            if args.model_type == "bert":
                token_type_ids = segment_ids
            else:
                token_type_ids = None

            if task_name in ["WiC", "WSC"]:
                span_1_mask, span_1_text, span_2_mask, span_2_text = batch[-4:]

                outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=input_mask,
                                span_1=span_1_mask, span_2=span_2_mask, labels=label_ids.float())
            else:
                outputs = model(task_id=task_id, input_ids=input_ids, token_type_ids=token_type_ids,
                                attention_mask=input_mask, labels=label_ids)

            logits = outputs[1].cpu()
            cur_loss = outputs[0]
            if n_gpu > 1:
                cur_loss = cur_loss.mean()

            if task_name in ["COPA", "WSC", "WiC"]:
                probs = sigmoid(logits).squeeze()
            else:
                probs = softmax(logits)

            if task_name in ["WSC", "WiC"]:
                preds = probs > 0.5
            else:
                preds = np.argmax(logits, axis=1).tolist()

            label_ids = label_ids.tolist()
            probs = probs.tolist()

            val_loss.append(cur_loss.item())

            if task_name in ["COPA"]:
                # multiple choice tasks, need to merge subinstance
                for ix, guid in enumerate(batch[0]):
                    val_preds[guid] = probs[ix]
                    val_labels[guid] = label_ids[ix]

                    if output_prediction:
                        ins_id, subins_id = guid.split("_")
                        if not ins_id in val_instance:
                            val_instance[ins_id] = dict()
                        val_instance[ins_id][subins_id] = (tokenizer.decode(input_ids[ix].tolist()),
                                                           label_ids[ix],
                                                           probs[ix])


            else:
                val_preds.extend(preds)
                val_labels.extend(label_ids)
                if output_prediction:
                    for ix, guid in enumerate(batch[0]):
                        cur_pred = int(preds[ix])
                        cur_label = label_ids[ix]
                        text = tokenizer.decode(input_ids[ix].tolist())
                        output = {"guid": guid,
                                  "pred": cur_pred,
                                  "label": cur_label,
                                  "text": [segment for segment in text if "[PAD]" not in segment],}

                        if isinstance(probs[0], list):
                            output["prob"] = [float(x) for x in probs[ix]]
                        else:
                            output["prob"] = probs[ix]

                        if task_name in ["WSC", "WiC"]:
                            output["span_1"] = span_1_text[ix]
                            output["span_2"] = span_2_text[ix]

                        fout.write(json.dumps(output) + "\n")


    if output_prediction:
        # for multiple choice cases, need to wait until a complete pass to
        # write to disk
        if task_name in ["COPA"]:
            for guid in val_instance:
                choice_1, choice_1_label, choice_1_prob = val_instance[guid]["0"]
                choice_2, choice_2_label, choice_2_prob = val_instance[guid]["1"]
                output = {"guid": guid,
                          "choice1": choice_1[0],
                          "choice2": choice_2[0],
                          "question": choice_1[1],
                          "premise": choice_1[2],
                          "choice1_prob": choice_1_prob,
                          "choice2_prob": choice_2_prob,
                          "label": "1" if choice_1_label else "2",
                          "prediction": "1" if choice_1_prob > choice_2_prob else "2"}
                fout.write(json.dumps(output) + "\n")

        fout.close()

    evaluator = evaluation_utils.TASK_TO_EVALUATOR[args.tasks[task_id]]
    metric = evaluator(val_preds, val_labels)
    return metric, np.mean(val_loss)

def run_evaluation(processed_test_samples, model, tokenizer, args, device, n_gpu):
    """Run evaluation for all tasks."""

    fout_log = open(args.output_dir + "/test_results.txt", 'w')
    for task_id, task in enumerate(args.tasks):
        task_val_acc, _ = _run_task_prediction(processed_task_samples=processed_test_samples[task],
                                               model=model, args=args, task_id=task_id, device=device,
                                               n_gpu=n_gpu, output_prediction=True, epoch_n=args.load_n_epoch,
                                               tokenizer=tokenizer)
        print("task: %s\tacc: %.4f" % (task, task_val_acc))
        fout_log.write("%s: %.4f" % (task, task_val_acc))
    fout_log.close()


def _merge_data_from_all_tasks(processed_samples, batch_size=1):
    """Merge data from different tasks into one list.

    Args:
        processed_samples: a dictionary mapping task names into datasets.
    Returns:
        merged_dataset: a list of training batches.
    """
    merged_dataset = []
    for task, task_data in processed_samples.items():
        cur_task_num_samples = len(task_data)
        cur_task_num_batches = cur_task_num_samples // batch_size
        for ix in range(cur_task_num_batches):
            cur_batch = task_data[ix * batch_size: (ix + 1) *batch_size]
            merged_dataset.append((task, cur_batch))
    return merged_dataset


def _calculate_logging_step(tasks, logging_freq, batch_size, task_tr_samples):
    """Calculate for each task how many steps to log information and write to tensorboard.

    Args:
         tasks: a list of task names
         logging_freq: int. How many loggings per epoch.
         batch_size:
         task_tr_samples: a dictionary mapping task name to training samples.
    """
    task_logging_steps = dict()
    for task in tasks:
        total_steps = task_tr_samples[task] // batch_size
        logging_steps = total_steps // logging_freq
        task_logging_steps[task] = logging_steps
    return task_logging_steps


def run_training(processed_train_samples, processed_val_samples, model, optimizer, scheduler,
                 tokenizer, args, device, n_gpu):
    """Run training for specified number of epochs.

    Args:
        processed_train_samples: a dictionary mapping task name to its corresponding training SuperGlueDataset
        processed_val_samples: a dictionary mapping task name to its corresponding validation SuperGlueDataset
    """
    tb_writer = SummaryWriter(logdir=os.path.join(args.output_dir, "tensorboard"), flush_secs=30)

    fout_log = open(args.output_dir + "/eval_results.txt", 'w')
    logger.info("validation before training...")

    fout_log.write("EPOCH\t\t")
    for task_id, task in enumerate(args.tasks):
        fout_log.write("%d:%s\t\t" % (task_id, task))
    fout_log.write("\n")

    fout_log.write("-1\t\t")

    for task_id, task in enumerate(args.tasks):
        task_val_acc, task_val_loss = _run_task_prediction(processed_task_samples=processed_val_samples[task],
                                                           model=model, args=args, task_id=task_id, device=device,
                                                           n_gpu=n_gpu, output_prediction=True,  epoch_n=-1,
                                                           tokenizer=tokenizer)

        fout_log.write("%.2f(%.3f)\t" % (task_val_acc * 100, task_val_loss))
        logger.info("Task: %s\tValidation acc: %.2f\tValidation loss: %.3f" % (task, task_val_acc * 100, task_val_loss))
    fout_log.write("\n")

    # for MTL training, merge all batches
    merged_training_data = _merge_data_from_all_tasks(processed_train_samples, args.batch_size)
    logger.info("****** Running training ******")
    logger.info("   Num examples total: %d" % (args.batch_size * len(merged_training_data)))

    if args.fp16:
        from apex import amp

    task_tr_loss = {task: 0 for task in args.tasks}
    task_logging_loss = {task: 0 for task in args.tasks}
    task_tr_steps = {task: 0 for task in args.tasks}
    task_tr_samples = {task: len(dataset) for task, dataset in processed_train_samples.items()}
    task_logging_steps = _calculate_logging_step(tasks=args.tasks, logging_freq=args.logging_freq,
                                                 batch_size=args.batch_size, task_tr_samples=task_tr_samples)
    softmax = nn.Softmax()

    for epoch_n in trange(int(args.num_train_epochs), desc="Epoch"):

        epoch_n += 1
        np.random.shuffle(merged_training_data)
        fout_train = open(args.output_dir + "/train_%d.jsonl" % epoch_n, 'w')

        for step, (task, batch) in enumerate(merged_training_data):

            model.train()
            task_id = args.tasks.index(task)
            batch_tensors = tuple(t.to(device) for t in batch[1:5])
            input_ids, input_mask, segment_ids, label_ids = batch_tensors

            if args.model_type == "bert":
                token_type_ids = segment_ids
            else:
                # roberta does not have segment id (token_type_id)
                token_type_ids = None

            if task in ["WiC", "WSC"]:
                span_1_mask, span_1_text, span_2_mask, span_2_text = batch[-4:]
                task_loss, _ = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=input_mask,
                                     span_1=span_1_mask, span_2=span_2_mask, labels=label_ids.float())
            else:
                task_loss, logits = model(task_id=task_id, input_ids=input_ids, token_type_ids=token_type_ids,
                                attention_mask=input_mask, labels=label_ids)

            logits = logits.detach().cpu()
            label_ids = label_ids.cpu().tolist()
            probs = softmax(logits)
            preds = np.argmax(logits, axis=-1).tolist()
            for ix, label in enumerate(label_ids):
                output_obj = {"guid": batch[0][ix]}
                text = tokenizer.decode(input_ids[ix].tolist())
                output_obj["text"] = text
                output_obj["label"] = label
                output_obj["probs"] = probs[ix].tolist()
                output_obj["pred"] = preds[ix]
                fout_train.write(json.dumps(output_obj) + "\n")

            if n_gpu > 1:
                task_loss = task_loss.mean()

            if args.fp16:
                with amp.scale_loss(task_loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                task_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            task_tr_loss[task] += task_loss.item()
            task_tr_steps[task] += 1

            scheduler.step() # update learning rate schedule
            optimizer.step()
            model.zero_grad()

            # run validation and log results to console and tensorboard
            if task_tr_steps[task] % task_logging_steps[task] == 0:

                avg_tr_loss = (task_tr_loss[task] - task_logging_loss[task]) / task_logging_steps[task]
                task_val_acc, task_val_loss = _run_task_prediction(
                                                        processed_task_samples=processed_val_samples[task],
                                                        model=model,
                                                        args=args,
                                                        task_id=task_id,
                                                        device=device,
                                                        n_gpu=n_gpu,
                                                        output_prediction=False,
                                                        epoch_n=epoch_n,
                                                        tokenizer=tokenizer)

                logger.info("Task: %s\tTrain loss: %.4f\tValid loss: %.4f\tValid acc: %.3f" % \
                            (task, avg_tr_loss, task_val_loss, task_val_acc * 100))

                tb_writer.add_scalars("loss_%s" % task,
                                      {"train_loss" : avg_tr_loss,
                                       "val_loss": task_val_loss},
                                      task_tr_steps[task])
                tb_writer.add_scalar("lr", scheduler.get_lr()[0], sum(task_tr_steps.values()))
                logger.info("learning rate: %s" % str(scheduler.get_lr()[0]))
                task_logging_loss[task] = task_tr_loss[task]
        fout_train.close()

        if not args.demo:
            output_model_file = os.path.join(args.output_dir, "epoch_%s" % epoch_n)
            model_to_save = model.module if hasattr(model, 'module') else model
            state = {"epoch": epoch_n, "model_state": model_to_save.state_dict(),
                     "optimizer_state": optimizer.state_dict(),}
            torch.save(state, output_model_file)


        fout_log.write("%d\t\t" % epoch_n)
        logger.info("Epoch %d finished" % epoch_n)
        for task_id, task in enumerate(args.tasks):
            task_val_acc, task_val_loss = _run_task_prediction(processed_task_samples=processed_val_samples[task],
                                                               model=model, args=args, task_id=task_id, device=device,
                                                               n_gpu=n_gpu, output_prediction=True, epoch_n=epoch_n,
                                                               tokenizer=tokenizer)

            fout_log.write("%.2f(%.3f)\t" % (task_val_acc * 100, task_val_loss))
            logger.info("Task: %s\tTraining loss: %.4f\tValidation acc: %.4f\tValidation loss: %.4f" \
                        % (task, task_tr_loss[task] / task_tr_steps[task], task_val_acc, task_val_loss))

        fout_log.write("\n")
        fout_log.flush()
    tb_writer.close()
    return
