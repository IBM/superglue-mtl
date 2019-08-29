# Author: Xinyu Hua
""" filter low quality sentences from the automatically rewritten set """

import json
import utils

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm, trange

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.tokenization import BertTokenizer
from bert_baseline_dataset_utils import convert_examples_to_features, InputExample

SET_TYPE = "train"
BERT_MODEL = "bert-large-uncased"
NUM_LABELS = 2
BATCH_SIZE = 128
MAX_SEQ_LENGTH = 128
FP16 = False
MODEL_PATH = utils.DATA_PATH + "model_dep/cola_bert-large_epoch_4.bin"


def create_dataloader():
    fpath = utils.DATA_PATH + "squad/squad_converted_statements.%s.jsonl" % SET_TYPE
    examples = []
    originals = []

    for ln in open(fpath):
        cur_obj = json.loads(ln)
        for item in cur_obj["qa"]:
            question_id = item["id"]
            examples.append(InputExample(guid=item["statement"][:20] + question_id,
                                         text_a=item["statement"],
                                         text_b=None,
                                         label=None))
            originals.append((item["statement"], item["id"], True))
            for l in item["lies"]:
                originals.append((l, item["id"], False))
                examples.append(InputExample(guid=l[:20] + question_id,
                                             text_a=l,
                                             text_b=None,
                                             label=None))

    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, do_lower_case=True)
    features = convert_examples_to_features(examples, None, MAX_SEQ_LENGTH, tokenizer, output_mode="test")

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    set_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)

    sampler = SequentialSampler(set_data)
    dataloader = DataLoader(set_data, sampler=sampler, batch_size=BATCH_SIZE)

    return dataloader, originals


def filter_low_quality():
    """ based on the filtering results generate a clean-up version of the dataset"""
    kept_data = dict()
    kept_id = set()
    for ln in open(utils.DATA_PATH + "squad/squad_converted_statements.%s.filtered.jsonl" % SET_TYPE):
        cur_obj = json.loads(ln)
        cur_id = cur_obj["id"]
        cur_statement = cur_obj["statement"]
        if cur_obj["prediction"] == 1:
            kept_id.add(cur_id)
            if not cur_id in kept_data:
                kept_data[cur_id] = []
            kept_data[cur_id].append(cur_statement)

    fout = open(utils.DATA_PATH + "squad/squad_converted_statements.%s.clean.jsonl" % SET_TYPE, 'w')
    for ln in open(utils.DATA_PATH + "squad/squad_converted_statements.%s.jsonl" % SET_TYPE):
        cur_obj = json.loads(ln)
        new_qa = []
        for qa in cur_obj["qa"]:
            cur_id = qa["id"]
            cur_qa = {"lies": []}
            if cur_id not in kept_id:continue
            for l in cur_obj["lies"]:
                if l in kept_data[cur_id]:
                    cur_qa["lies"].append(l)
            if cur_obj["statement"] in kept_data[cur_id]:
                cur_qa["statement"] = cur_obj["statement"]
            if len(cur_qa["lies"]) == 0 and "statement" not in cur_qa: continue
            cur_qa["id"] = cur_id
            new_qa.append(cur_qa)
        if len(new_qa) == 0: continue
        new_obj = {"title": cur_obj["title"],
                   "context": cur_obj["context"],
                   "qa": new_qa}
        fout.write(json.dumps(new_obj) + "\n")
    fout.close()


def generate_filter_labels():
    """ use CoLA trained BERT sequence classifier to remove low quality ones """

    model = BertForSequenceClassification.from_pretrained(BERT_MODEL,
                                                          cache_dir=PYTORCH_PRETRAINED_BERT_CACHE,
                                                          num_labels=NUM_LABELS)

    device = torch.device("cuda")
    n_gpu = torch.cuda.device_count()

    if FP16:
        model.half()
    model.to(device)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(device)

    eval_dataloader, originals = create_dataloader()
    print("data loaded, %d instance to process" % len(originals))
    model.eval()

    fout = open(utils.DATA_PATH + "squad/squad_converted_statements.%s.filtered.jsonl" % SET_TYPE, "w")
    softmax = nn.Softmax(dim=-1)
    nb_eval_steps = 0
    nb_filtered = 0
    nb_total = 0

    for input_ids, input_mask, segment_ids in tqdm(eval_dataloader):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)

        cur_original_list = originals[nb_eval_steps * BATCH_SIZE : nb_eval_steps * BATCH_SIZE + BATCH_SIZE]

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask)
            probs = softmax(logits)

        logits = logits.detach().cpu().numpy()
        probs = probs.detach().cpu().numpy()

        cur_preds = np.argmax(logits, axis=1)

        nb_eval_steps += 1
        for sample_ix in range(len(input_ids)):
            cur_pred = int(cur_preds[sample_ix])
            if cur_pred == 0: nb_filtered += 1
            nb_total += 1
            cur_original = cur_original_list[sample_ix]
            cur_probs = list(probs[sample_ix])
            fout.write(json.dumps({"id": cur_original[1], "statement": cur_original[0], "value": cur_original[2],
                                   "prediction": cur_pred, "prob": [str(x) for x in cur_probs]}) + "\n")

    fout.close()
    print("Acceptability test based filtering finished. %d out of %d are removed" % (nb_filtered, nb_total))
    return


def make_final_dataset():
    """
    Based on filter information construct the final trainable dataset
    """
    filter_info = utils.load_squad_rewritten_filter_info(set_type=SET_TYPE)
    rewritten_raw = utils.load_squad_rewritten(set_type=SET_TYPE)
    fout = open(utils.DATA_PATH + "squad/squad_converted_final.%s.jsonl" % SET_TYPE, 'w')

    filter_dict = dict()
    for ln in filter_info:
        if ln["prediction"] != 1:continue
        cur_id = ln["id"]
        if not cur_id in filter_dict:
            filter_dict[cur_id] = []
        filter_dict[cur_id].append(ln["statement"])

    pair_cnt = 0
    for ln in rewritten_raw:
        cur_obj = {"title": ln["title"], "context": ln["context"], "statements": []}
        for qa in ln["qa"]:
            cur_id = qa["id"]
            if not cur_id in filter_dict:continue
            if qa["statement"] in filter_dict[cur_id]:
                cur_obj["statements"].append((cur_id, qa["statement"], True))
            for lie in qa["lies"]:
                if lie in filter_dict[cur_id]:
                    cur_obj["statements"].append((cur_id, lie, False))
        if len(cur_obj["statements"]) == 0:continue
        pair_cnt += len(cur_obj["statements"])
        fout.write(json.dumps(cur_obj) + "\n")
    fout.close()
    print("%d pairs in total" % pair_cnt)


if __name__ == '__main__':
    # generate_filter_labels()
    # filter_low_quality()
    make_final_dataset()
