# Author: Xinyu Hua
""" converting question into statements on boolQ or SQuAD """

import json
import collections
import file_utils as utils
from rewritting_utils import *

def convert_squad_questions():
    """
    convert squad questions into statements
    """
    set_type = "dev"

    data = utils.load_squad_context(set_type)
    ctx_parsed = utils.load_squad_context_parsing_info(set_type)
    question_parsed = utils.load_squad_question_parsing_info(set_type)
    print("%s data loaded" % set_type)

    rule_counter = collections.Counter()
    total_considered = 0
    num_matched = 0
    fout = open(utils.DATA_PATH + "squad/squad_converted_statements.%s.jsonl" % set_type, "w")

    for ctx in data:
        context = ctx["context"]
        context_parsed = ctx_parsed[context]
        entity_mentions = []
        for sent in context_parsed["sentences"]:
            for em in sent["entitymentions"]:
                entity_mentions.append((em["text"], em["ner"]))

        title = ctx["title"]
        cur_obj = {"title": title, "context": context, "qa": []}
        for question, answer, id in ctx["qa"]:
            parse = question_parsed[id]
            tokens = parse["tokens"]
            const_parse = read_const_parse(parse["parse"])
            answer_txt = answer["text"]
            total_considered += 1

            for rule in CONVERSION_RULES:
                sent, neg_sent = rule.convert(question, answer_txt, tokens, const_parse, entity_mentions)
                if sent:
                    # print("question: %s" % question.encode("utf-8"))
                    # print("statement: %s" % sent.encode("utf-8"))
                    # for ns_id, ns in enumerate(neg_sent):
                    #     print("lie (%d): %s" % (ns_id + 1, ns.encode('utf-8')))
                    # print("rule: %s\n" % rule.name)
                    rule_counter[rule.name] += 1
                    num_matched += 1
                    cur_obj["qa"].append({"question": question, "answer": answer_txt, "statement": sent, "lies": neg_sent, "id": id})
                    break
            if len(cur_obj["qa"]) == 0: continue
        fout.write(json.dumps(cur_obj) + "\n")
    fout.close()

    print('=== Summary ===')
    print('Matched %d/%d = %.2f%% questions' % (
        num_matched, total_considered, 100.0 * num_matched / total_considered))
    for rule in CONVERSION_RULES:
        num = rule_counter[rule.name]
        print('  Rule "%s" used %d times = %.2f%%' % (
            rule.name, num, 100.0 * num / total_considered))


def generate_examples():
    """
    generating sample context vs rewritten questions
    """
    rewritten = utils.load_squad_rewritten()
    fout = open("squad_rewritten.txt", 'w')
    for cid, ln in enumerate(rewritten):
        fout.write("TITLE: %s\n" % ln["title"].encode("utf-8"))
        fout.write("CONTEXT (%d): %s\n\n" % (cid + 1, ln["context"].encode("utf-8")))
        for qid, q in enumerate(ln["qa"]):
            fout.write("QUESTION (%d): %s\n" % (qid + 1, q["question"].encode("utf-8")))
            fout.write("ANSWER: %s\n" % q["answer"].encode("utf-8"))
            fout.write("STATEMENT: %s\n\n" % q["statement"].encode("utf-8"))
        if cid == 100:break
    fout.close()



if __name__ == "__main__":
    convert_squad_questions()
