import os
import json
import logging
import torch
import numpy as np
from torch.utils.data import TensorDataset, Dataset

logging.getLogger().setLevel(logging.INFO)

DATA_PATH = os.environ["SG_DATA"]
EXP_PATH = os.environ["SG_MTL_EXP"]


BERT_LARGE_MNLI_PATH = "/datastor/xhua/Experiments/mnli_bert_large_no_transfer_seed_42_lr_1e-5_max_seq_len_256/pytorch_model.bin_epoch_4_train_loss_0.0049_dev_loss_0.5259_acc_86.52"
BERT_LARGE_SQUAD_NLI_PATH = "/dccstor/xhua11/Experiments/squadnli_bert-large_transfer-wwm_lr-1e-5_seed-42_max-len-256/pytorch_model.bin_epoch_1_train_loss_0.2614_val_loss_0.2474"
BERT_LARGE_SQUAD_1_PATH = "/u/avi/Projects/dccstor_avi5/nq/trained_squad_models/using_mglass_pretraining/bert_large_sq1.bin"
BERT_LARGE_SQUAD_2_PATH = "/u/avi/Projects/dccstor_avi5/nq/trained_squad_models/bert_large_sq2_mglass_pretrained.bin"
BERT_LARGE_WWM_SQUAD_2_PATH = "/dccstor/panlin2/squad2/expts/Pan_squad2_whole_word_32bs/output/pytorch_model.bin"

TRANSFER_PATH = {
    "squad1-bert-large" : BERT_LARGE_SQUAD_1_PATH,
    "squad2-bert-large" : BERT_LARGE_SQUAD_2_PATH,
    "squad2-bert-large-wwm" : BERT_LARGE_WWM_SQUAD_2_PATH,
    "squad2nli-roberta-large" : ROBERTA_LARGE_SQUAD_NLI_PATH,
}


class InputExample(object):
    """Base class for a single input example, this will be inherited for specific tasks."""

    def __init__(self, guid, text_hyp, text_pre=None, label=None):
        """Constructs an InputExample.

        Args:
             guid: Unique id for the example.
             text_hyp: string. The untokenized text of the hypothesis.
             text_pre: (Optional) string. The untokenized text of the premise. This is empty for
                tasks that have only one text unit in the input.
             label: (Optional) string. The label of the example. This should be None for test set,
                and needs to be specified for train/val set.
        """
        self.guid = guid
        self.text_hyp = text_hyp
        self.text_pre = text_pre
        self.label = label

    def featurize_example(self, *kargs, **kwargs):
        raise NotImplementedError


class DefaultInputExample(InputExample):
    """Default input example class, used for sequence classification tasks with one or two text units in input."""

    def __init__(self, guid, text_hyp, text_pre, label):
        super(DefaultInputExample, self).__init__(guid, text_hyp, text_pre, label)

    def featurize_example(self, tokenizer, max_seq_length=128, label_map=None, output_mode="classification",
                          model_type="bert", print_example=False, task=None):
        """Tokenize example into word ids and masks.

           Args:
               tokenizer: either a BertTokenizer or a RobertaTokenizer
               max_seq_length: int. The maximum allowed number of bpe units for the input.
               label_map: dictionary. A map that returns the label_id given the label string.
               model_type: string. Either `bert` or `roberta`. For `roberta` there will be an extra sep token in
                    the middle.

           The default behavior is:
           tokens: [tokenizer.cls_token] + self.text_hyp + [tokenizer.sep_token] + self.text_pre + [tokenizer.sep_token]
           segment_ids: 0                      0...0             0                    1...1              1

           For tasks without self.text_pre, the tokenization will be:
           tokens: [tokenizer.cls_token] + self.text_hyp + [tokenizer.sep_token]
        """

        tokens_a = tokenizer.tokenize(self.text_hyp)
        if self.text_pre:
            tokens_b = tokenizer.tokenize(self.text_pre)

            special_tokens_count = 4 if model_type == "roberta" else 3
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)
        else:
            special_tokens_count = 3 if model_type == "roberta" else 2
            if len(tokens_a) > max_seq_length - special_tokens_count:
                tokens_a = tokens_a[:max_seq_length - special_tokens_count]

        tokens = tokens_a + [tokenizer.sep_token]
        if model_type == "roberta":
            tokens += [tokenizer.sep_token]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [tokenizer.sep_token]
            segment_ids += [1] * (len(tokens_b) + 1)

        tokens = [tokenizer.cls_token] + tokens
        segment_ids = [0] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        padding_length = max_seq_length - len(input_ids)
        input_ids = input_ids + tokenizer.convert_tokens_to_ids([tokenizer.pad_token] * padding_length)
        input_mask = input_mask + [0] * padding_length
        segment_ids = segment_ids + [0] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[self.label]
        elif output_mode == "regression":
            label_id = float(self.label)
        else:
            raise KeyError

        if print_example:
            logging.info("*** Example (%s) ***" % task)
            logging.info("guid: %s" % (self.guid))
            logging.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logging.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logging.info("label: %s (id = %s)" % (str(self.label), str(label_id)))

        return InputFeatures(input_ids=input_ids,
                             input_mask=input_mask,
                             segment_ids=segment_ids,
                             label_id=label_id)


class WSCInputExample(InputExample):
    """
    InputExample for WSC data, where each instance is a piece of text with two spans, the goal is to classify
    whether the two spans refer to the same entity.
    """

    def __init__(self, guid, text, span_1, span_2, label):
        super(WSCInputExample, self).__init__(guid, text_hyp=text, text_pre=None, label=label)
        self.spans = [span_1, span_2]

    def featurize_example(self, tokenizer, max_seq_length=128, label_map=None, model_type="bert", print_example=False,
                          task=None):
        """Tokenize example for WSC.
        Args:
            tokenizer: either a BertTokenizer or a RobertaTokenizer
            max_seq_length: int. The maximum allowed number of bpe units for the input.
            label_map: dictionary. A map that returns the label_id given the label string.
            model_type: string. Either `bert` or `roberta`. For `roberta` there will be an extra sep token in
                    the middle.
            print_example: bool. If set to True, print the tokenization information for current instance.
        """
        tokens_a = tokenizer.tokenize(self.text_hyp)
        token_word_ids = _get_word_ids(tokens_a, model_type)
        span_1_tok_ids = _get_token_ids(token_word_ids, self.spans[0][0], offset=1)
        span_2_tok_ids = _get_token_ids(token_word_ids, self.spans[1][0], offset=1)
        # span_1_tok_ids: list(int), such as [2,3,4]

        special_tokens_count = 2
        if len(tokens_a) > max_seq_length - special_tokens_count:
            tokens_a = tokens_a[:max_seq_length - special_tokens_count]

        tokens = tokens_a + [tokenizer.sep_token]
        segment_ids = [0] * len(tokens)

        tokens = [tokenizer.cls_token] + tokens
        segment_ids = [0] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        padding_length = max_seq_length - len(input_ids)
        input_ids = input_ids + tokenizer.convert_tokens_to_ids([tokenizer.pad_token] * padding_length)
        input_mask = input_mask + [0] * padding_length
        segment_ids = segment_ids + [0] * padding_length

        span_1_mask = [0] * len(input_ids)
        for k in span_1_tok_ids:
            span_1_mask[k] = 1

        span_2_mask = [0] * len(input_ids)
        for k in span_2_tok_ids:
            span_2_mask[k] = 1

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(span_1_mask) == max_seq_length
        assert len(span_2_mask) == max_seq_length


        if self.label is not None:
            label_id = int(self.label)
        else:
            label_id = None


        if print_example:
            logging.info("*** Example (%s) ***" % task)
            logging.info("guid: %s" % (self.guid))
            logging.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logging.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logging.info("label: %s (id = %s)" % (str(self.label), str(label_id)))

        return InputFeatures(input_ids=input_ids,
                             input_mask=input_mask,
                             segment_ids=segment_ids,
                             span_1_mask=span_1_mask,
                             span_1_text=self.spans[0][1],
                             span_2_mask=span_2_mask,
                             span_2_text=self.spans[1][1],
                             label_id=label_id)


class COPAInputExample(InputExample):
    """The input example class for COPA."""

    def __init__(self, guid, text_pre, text_choice_1, text_choice_2, question, label=None):
        """Constrcuts a COPAInputExample.

        Args:
            guid: Unique id for the example.
            text_pre: string. The untokenized text of the premise.
            text_choice_1: string. The untokenized text of choice 1.
            text_choice_2: string. The untokenized text of choice 2.
            question: string. `cause` or `effect`
            label: (Optional) int. The label for the example, either 0 or 1.
        """
        super(COPAInputExample, self).__init__(guid=guid, text_hyp=None, text_pre=text_pre, label=label)
        self.text_choice_1 = text_choice_1
        self.text_choice_2 = text_choice_2
        self.question = question

    def featurize_example(self, tokenizer, max_seq_length=128, label_map=None, model_type="bert", print_example=False,
                          task=None):
        """
        Tokenize example for COPA. Each training instance will result in two examples.
        Args:
               tokenizer: either a BertTokenizer or a RobertaTokenizer
               max_seq_length: int. The maximum allowed number of bpe units for the input.
               label_map: dictionary. A map that returns the label_id given the label string.
               model_type: string. Either `bert` or `roberta`. For `roberta` there will be an extra sep token in
                    the middle.

           For COPA, one instance will be used to construct the following two examples:
           tokens_1: [tokenizer.cls_token] + self.text_choice_1 + [tokenizer.sep_token] +
                    self.question + [tokenizer.sep_token] + self.text_pre + [tokenizer.sep_token]
           tokens_2: [tokenizer.cls_token] + self.text_choice_2 + [tokenizer.sep_token] +
                    self.question + [tokenizer.sep_token] + self.text_pre + [tokenizer.sep_token]
        """

        def _featurize_example(text_a, text_b, text_c, cur_label=None, print_example=False):
            tokens_a = tokenizer.tokenize(text_a)
            tokens_b = tokenizer.tokenize(text_b)
            tokens_c = tokenizer.tokenize(text_c)
            special_tokens_count = 6 if model_type == "roberta" else 4
            _truncate_seq_pair(tokens_a, tokens_c, max_seq_length - special_tokens_count - len(tokens_b))
            tokens = tokens_a + [tokenizer.sep_token]
            if model_type == "roberta":
                tokens += [tokenizer.sep_token]
            segment_ids = [0] * len(tokens)

            tokens += tokens_b + [tokenizer.sep_token]
            segment_ids += [1] * (len(tokens_b) + 1)
            if model_type == "roberta":
                tokens += [tokenizer.sep_token]
                segment_ids += [1]

            tokens += tokens_c + [tokenizer.sep_token]
            segment_ids += [2] * (len(tokens_c) + 1)

            tokens = [tokenizer.cls_token] + tokens
            segment_ids = [0] + segment_ids

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            padding_length = max_seq_length - len(input_ids)
            input_ids = input_ids + tokenizer.convert_tokens_to_ids([tokenizer.pad_token] * padding_length)
            input_mask = input_mask + [0] * padding_length
            segment_ids = segment_ids + [0] * padding_length

            label_id = float(cur_label) if cur_label is not None else None

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            if print_example:
                logging.info("*** Example (COPA) ***")
                logging.info("guid: %s" % (self.guid))
                logging.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
                logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logging.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                logging.info("label: %s (id = %s)" % (str(cur_label), str(label_id)))

            return InputFeatures(input_ids=input_ids,
                                 input_mask=input_mask,
                                 segment_ids=segment_ids,
                                 label_id=label_id)

        feat_ex_1 = _featurize_example(self.text_choice_1,
                                       self.question,
                                       self.text_pre,
                                       cur_label=int(self.label == 0),
                                       print_example=print_example)
        feat_ex_2 = _featurize_example(self.text_choice_2,
                                       self.question,
                                       self.text_pre,
                                       cur_label=int(self.label == 1),
                                       print_example=print_example)
        return feat_ex_1, feat_ex_2


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, **kwargs):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.__dict__.update(kwargs)


class SuperGlueDataset(Dataset):

    def __init__(self, task_loader, task_samples, tokenizer, max_seq_length, model_type="bert"):
        label_map = {label: i for i, label in enumerate(task_loader.get_labels())}
        label_map[None] = None

        features = []
        self.all_guids = []
        self.task_name = task_loader.task_name
        for (ex_index, example) in enumerate(task_samples):
            print_example = True if ex_index < 1 else False
            featurized_example = example.featurize_example(tokenizer,
                                                           max_seq_length=max_seq_length,
                                                           label_map=label_map,
                                                           model_type=model_type,
                                                           print_example=print_example,
                                                           task=task_loader.task_name)
            if task_loader.task_name == "COPA":
                features.append(featurized_example[0])
                features.append(featurized_example[1])
                self.all_guids.append(str(example.guid) + "_0")
                self.all_guids.append(str(example.guid) + "_1")
            else:
                features.append(featurized_example)
                self.all_guids.append(str(example.guid))


        self.all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        self.all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        self.all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

        if task_loader.task_name in ["WSC", "WiC"]:
            self.all_span_1_mask = torch.tensor([f.span_1_mask for f in features], dtype=torch.long)
            self.all_span_1_text = [f.span_1_text for f in features]
            self.all_span_2_mask = torch.tensor([f.span_2_mask for f in features], dtype=torch.long)
            self.all_span_2_text = [f.span_2_text for f in features]
       
        # if label is unavailable
        if features[0].label_id is None:
            self.all_label_ids = torch.tensor([0 for _ in features], dtype=torch.long)
        elif task_loader.task_name == "COPA":
            self.all_label_ids = torch.tensor([f.label_id for f in features],
                                     dtype=torch.float)
        else:
            self.all_label_ids = torch.tensor([f.label_id for f in features],
                                              dtype=torch.long)

    def __len__(self):
        return len(self.all_guids)

    def __getitem__(self, index):
        item = (self.all_guids[index], self.all_input_ids[index], self.all_input_mask[index],
                     self.all_segment_ids[index], self.all_label_ids[index])
        if self.task_name in ["WSC", "WiC"]:
            item = item + (self.all_span_1_mask[index], self.all_span_1_text[index],
                           self.all_span_2_mask[index], self.all_span_2_text[index])

        return item


def load_jsonl_raw(dataset, demo=False, set_type="train"):
    fname = "%s.jsonl" % set_type
    lines = []
    for ln in open(os.path.join(DATA_PATH, dataset, fname)):
        lines.append(json.loads(ln))
        if demo and len(lines) >= 500:
            break
    return lines


def load_squadnli_raw(demo=False, set_type="train"):
    """Load SQuADNLI data from file.

    Return:
         lines: each line is a dictionary with the following fields:
            `premise`: string. The context paragraph.
            `hypothesis`: string. Rewritten statement.
            `label`:  bool. Either true (entailed) or false (not entailed).
            `id`: global unique id
    """
    fname = "squadnli.%s.jsonl" % set_type
    lines = []
    for ln in open(os.path.join(DATA_PATH + "../../squadnli/", fname)):
        cur_obj = json.loads(ln)
        context = cur_obj["context"]
        for item in cur_obj["statements"]:
            lines.append({"premise": context, "hypothesis": item[1], "label": item[-1], "id": item[0]})
            if demo and len(lines) >= 100:
                break
    return lines


class DataLoader(object):
    task_name = None

    def get_train_examples(self, demo=False):
        return self._create_examples(load_jsonl_raw(dataset=self.task_name,
                                                    demo=demo,
                                                    set_type="train"))

    def get_test_examples(self, demo=False, test_set="val"):
        return self._create_examples(load_jsonl_raw(dataset=self.task_name,
                                                    demo=demo,
                                                    set_type=test_set))

class SQuADNLILoader(DataLoader):
    task_name = "SQuADNLI"

    def get_train_examples(self, demo=False):
        return self._create_examples(load_squadnli_raw(demo=demo, set_type="train"))

    def get_test_examples(self, demo=False, test_set="val"):
        return self._create_examples(load_squadnli_raw(demo=demo, set_type=test_set))

    def get_labels(self):
        return [False, True]

    def _create_examples(self, lines):
        examples = []
        for (i, line) in enumerate(lines):
            text_hyp = line["hypothesis"]
            text_pre = line["premise"]
            if "label" in line:
                label = line["label"]
            else:
                label = None

            examples.append(
                DefaultInputExample(guid=line["id"],
                                    text_hyp=text_hyp,
                                    text_pre=text_pre,
                                    label=label)
            )
        return examples


class BoolQLoader(DataLoader):
    task_name = "BoolQ"

    def get_labels(self):
        return [False, True]

    def _create_examples(self, lines):
        examples = []
        for (i, line) in enumerate(lines):
            # in boolQ, the hypothesis is question, the premise is the passage
            text_hyp = line["question"]
            text_pre = line["passage"]
            if "label" in line:
                label = line["label"]
            else:
                label = None

            examples.append(
                DefaultInputExample(guid=line["idx"],
                                    text_hyp=text_hyp,
                                    text_pre=text_pre,
                                    label=label)
            )
        return examples


class BoolQNLILoader(BoolQLoader):
    task_name = "BoolQNLI"

    def get_train_examples(self, demo=False):
        return self._create_examples(load_jsonl_raw(dataset="BoolQNLI",
                                                    demo=demo,
                                                    set_type="train"))

    def get_test_examples(self, demo=False, test_set="val"):
        return self._create_examples(load_jsonl_raw(dataset="BoolQNLI",
                                                    demo=demo,
                                                    set_type=test_set))

    def _create_examples(self, lines):
        examples = []
        for (i, line) in enumerate(lines):
            # in boolQ, the hypothesis is question, the premise is the passage
            text_hyp = line["question"]
            text_pre = line["passage"]
            if "label" in line:
                label = line["label"]
            else:
                label = None

            examples.append(
                DefaultInputExample(guid=str(i),
                                    text_hyp=text_hyp,
                                    text_pre=text_pre,
                                    label=label)
            )
        return examples

class RTELoader(DataLoader):
    task_name = "RTE"

    def get_train_examples(self, demo=False):
        return self._create_examples(load_jsonl_raw(dataset="RTE",
                                                    demo=demo,
                                                    set_type="train"))

    def get_test_examples(self, demo=False, test_set="val"):
        return self._create_examples(load_jsonl_raw(dataset="RTE",
                                                    demo=demo,
                                                    set_type=test_set))

    def get_labels(self):
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines):
        examples = []
        for (i, line) in enumerate(lines):
            text_hyp = line["hypothesis"]
            text_pre = line["premise"]
            guid = line["idx"]
            if "label" in line:
                label = line["label"]
            else:
                label = None
            examples.append(
                DefaultInputExample(guid=guid,
                                    text_hyp=text_hyp,
                                    text_pre=text_pre,
                                    label=label)
            )
        return examples


class CBLoader(DataLoader):
    task_name = "CB"

    def get_train_examples(self, demo=False):
        return self._create_examples(load_jsonl_raw(dataset="CB",
                                                    demo=demo,
                                                    set_type="train"))

    def get_test_examples(self, demo=False, test_set="val"):
        return self._create_examples(load_jsonl_raw(dataset="CB",
                                                    demo=demo,
                                                    set_type=test_set))

    def get_labels(selfself):
        return ["entailment", "neutral", "contradiction"]

    def _create_examples(self, lines):
        examples = []
        for (i, line) in enumerate(lines):
            text_hyp = line["hypothesis"]
            text_pre = line["premise"]
            guid = line["idx"]
            if "label" in line:
                label = line["label"]
            else:
                label = None

            examples.append(
                DefaultInputExample(guid=guid,
                                    text_hyp=text_hyp,
                                    text_pre=text_pre,
                                    label=label)
            )
        return examples


class WSCLoader(DataLoader):
    task_name = "WSC"

    def get_labels(self):
        return [0]

    def _create_examples(self, lines):
        examples = []
        for (i, line) in enumerate(lines):
            text = line["text"]
            span_1 = (line["target"]["span1_index"], line["target"]["span1_text"])
            span_2 = (line["target"]["span2_index"], line["target"]["span2_text"])

            if "label" in line:
                label = line["label"]
            else:
                label = None

            examples.append(
                WSCInputExample(guid=line["idx"],
                                text=text,
                                span_1=span_1,
                                span_2=span_2,
                                label=label)
            )

        return examples


class COPALoader(DataLoader):
    task_name = "COPA"

    def get_labels(self):
        """this is set to [0] because we treat each subinstance as a logistic regression task"""
        return [0]

    def _create_examples(self, lines):
        examples = []
        for (i, line) in enumerate(lines):
            # following jiant's conversion of question
            question = line["question"]
            question = (
                "What was the cause of this?"
                if question == "cause"
                else "What happened as a result?"
            )

            if "label" in line:
                label = line["label"]
            else:
                label = None
            examples.append(
                COPAInputExample(guid=line["idx"],
                                 text_pre=line["premise"],
                                 text_choice_1=line["choice1"],
                                 text_choice_2=line["choice2"],
                                 question=question,
                                 label=label)
            )

        return examples


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def _get_span_tokens(tokenizer, text, spans):
    span1 = spans["span1_index"]
    span1_text = spans["span1_text"]
    span2 = spans["span2_index"]
    span2_text = spans["span2_text"]

    # construct end spans given span text space-tokenized length
    span1 = [span1, span1 + len(span1_text.strip().split())]
    span2 = [span2, span2 + len(span2_text.strip().split())]
    indices = [span1, span2]
    sorted_indices = sorted(indices, key=lambda x: x[0])

    # find span indices and text
    current_tokenization = []
    span_mapping = {}

    text_tokens = text.split()

    # align first span to tokenized text
    new_tokens = tokenizer.tokenize(" ".join(text_tokens[: sorted_indices[0][0]]))
    current_tokenization.extend(new_tokens)
    new_span1start = len(current_tokenization)
    span_tokens = tokenizer.tokenize(" ".join(text_tokens[sorted_indices[0][0]: sorted_indices[0][1]]))
    current_tokenization.extend(span_tokens)
    new_span1end = len(current_tokenization)
    span_mapping[sorted_indices[0][0]] = [new_span1start + 1, new_span1end + 1]

    # re-indexing second span
    new_tokens = tokenizer.tokenize(" ".join(text_tokens[sorted_indices[0][1]: sorted_indices[1][0]]))
    current_tokenization.extend(new_tokens)
    new_span2start = len(current_tokenization)
    span_tokens = tokenizer.tokenize(" ".join(text_tokens[sorted_indices[1][0]: sorted_indices[1][1]]))
    current_tokenization.extend(span_tokens)
    new_span2end = len(current_tokenization)
    span_mapping[sorted_indices[1][0]] = [new_span2start + 1, new_span2end + 1]

    return [span_mapping[spans["span1_index"]], span_mapping[spans["span2_index"]]]


def _get_token_ids(token_word_ids, span_word_id, offset=1):
    """Retrieve token ids based on word ids.

    Args:
        token_word_ids: the list of word ids for token.
        span_word_id: int. the word id in the original string.
        offset: int. if the tokenized sequence is prepended with special token, this offset will be set to
        the number of special tokens (for example, if [CLS] is added, then offset=1).

    For example, the token word ids can be:
     ['ir', 'an', 'Ġand', 'Ġaf', 'ghan', 'istan', 'Ġspeak', 'Ġthe', 'Ġsame', 'Ġlanguage', 'Ġ.']
    And the original sentence is "iran and afghanistan speak the same language ."
    Suppose the span_word_id is 2 (afghanistan), then the token id is [3, 4, 5]
    """
    results = []
    for ix, word_id in enumerate(token_word_ids):
        if word_id == span_word_id:
            results.append(ix + offset)
        elif word_id > span_word_id:
            break
    return results


def _get_word_ids(tokens, model_type="bert"):
    """Given the BPE split results, mark each token with its original word ids.

    Args:
          tokens: a list of BPE units
    For example, if original sentnece is `iran and afghanistan speak the same language .`, then the roberta
    tokens will be:
    ['ir', 'an', 'Ġand', 'Ġaf', 'ghan', 'istan', 'Ġspeak', 'Ġthe', 'Ġsame', 'Ġlanguage', 'Ġ.']
    The word ids will be:
    [0,     0,     1,     2,    2,      2,        3,        4,      5,      6,      7]

    Note: this method assume the original sentence is split by one space and is already tokenized.
    """

    word_ids = []
    for tok in tokens:
        if len(word_ids) == 0:
            word_ids.append(0)
            continue

        if "roberta" in model_type:
            if tok[0] != "Ġ":
                word_ids.append(word_ids[-1])
            else:
                word_ids.append(word_ids[-1] + 1)
        else:
            if tok[:1] == "##":
                word_ids.append(word_ids[-1])
            else:
                word_ids.append(word_ids[-1] + 1)
    return word_ids

TASK_TO_LOADER = {
    "BoolQ": BoolQLoader,
    "BoolQNLI": BoolQNLILoader,
    "SQuADNLI": SQuADNLILoader,
    "RTE": RTELoader,
    "CB": CBLoader,
    "COPA": COPALoader,
    "WSC": WSCLoader,
}
