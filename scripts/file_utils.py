# Author: Xinyu Hua
"""
utility functions to load data from disks

Note: all SQuAD data in this file is on SQuAD-2.0
"""

import json
import os

DATA_PATH = os.environ["SG_DATA"]
EXP_PATH = os.environ["SG_MTL_EXP"]



def load_squad_context(set_type="train"):
    """
    Load SQuAD data, where the data is a list of q/a pairs for each context.

    Note: assumes squad data stored in `{DATA_PATH}/squad/` in the format
    of `train-v2.0.json`, `dev-v2.0.json`

    Args:
        set_type: string. Either `train` or `dev`
    Returns:
        context_list: a list of articles, where each article contains multiple
            paragraphs, each paragraph has a list of q/a pairs. Each q/a pair
            is represented as a tuple of (question, answer_str, id)
    """
    file_path = DATA_PATH + "squad/%s-v2.0.json" % set_type

    with open(file_path) as jf:
        data = json.load(jf)

    data = data['data']

    context_list = []
    context_count = 0
    for line in data:
        title = ln["title"]
        context_count += len(ln["paragraphs"])

        for ctx in line["paragraphs"]:
            context = ctx["context"]
            cur_obj = {"title": ln["title"],
                       "context": context,
                       "qa": []}

            for qas in ctx["qas"]:
                # do not consider unanswerable
                if qas["is_impossible"]: 
                    continue

                # when multiple answers available, pick the top one only
                # (ranked by length of text)
                ans = sorted(qas["answer"], key=lambda x: len(x["text"]))
                cur_obj["qa"].append((qas["question"], ans[0], qas["id"]))
            context_list.append(cur_obj)
    print("%d articles loaded" % len(context_list))
    print("%d context (paragraphs) loaded" % context_count)
    return context_list
                

def load_squad_question_parsing_info(set_type="train"):
    """
    Load parsing information for questions in SQuAD-2.0.
    """

    file_path = DATA_PATH + "squad/%s.parse.jsonl" % set_type
    question_id2parse = dict()

    for line in open(file_path):
        cur_obj = json.loads(ln)
        question_id2parse[cur_obj["id"]] = cur_obj["question_parse"]
    print("question parsing information loaded")
    return question_id2parse


def load_squad_context_parsing_info(set_type="train"):
    """
    Load parsing information for context (paragraphs) in SQuAD-2.0.
    """
    ctx2parse = dict()
    file_path = DATA_PATH + "squad/%s.ctx.parse.jsonl" % set_type
    for ln in open(file_path):
        cur_obj = json.loads(ln)
        ctx2parse[cur_obj["context"]] = cur_obj["context_parse"]
    print("context parsing information loaded")

def load_squad_rewritten(set_type="train"):
    """
    Load rewritten version of SQuAD
    """
    file_path = DATA_PATH + "squad/squad_converted_statements.%s.jsonl" % set_type
    return [json.loads(ln) for ln in open(file_path)]
