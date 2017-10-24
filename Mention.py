#coding=utf8
import json
import sys
import numpy

from conf import *

class Mention():
    # mention id 是在gold里的id对应
    # mention num 是在自己篇章中 按顺序来的
    def __init__(self,mention_dict,language="cn"):
        self.dep_relation = mention_dict["dep_relation"]
        self.sent_num = mention_dict["sent_num"]
        self.head_index = mention_dict["head_index"]
        self.doc_id = mention_dict["doc_id"]
        self.contained_in_other_mention = mention_dict["contained-in-other-mention"]
        self.mention_type = mention_dict["mention_type"]

        self.mention_num = mention_dict["mention_num"] ## just an index
        self.mention_id = mention_dict["mention_id"] ## id appears in gold file

        self.start_index = mention_dict["start_index"]
        self.end_index = mention_dict["end_index"]

        if language == "cn":
            self.sentence = [x.encode('utf-8') for x in mention_dict["sentence"]]
            self.dep_parent = mention_dict["dep_parent"].encode("utf8")
        else:
            self.sentence = mention_dict["sentence"]
            self.dep_parent = mention_dict["dep_parent"]

        self.words = self.sentence[self.start_index:self.end_index]

class Document():
    def __init__(self,json_dict,gold_chain,language="cn"):
        ## document_features
        doc_features = json_dict["document_features"]
        self.doc_type = doc_features["type"]
        self.doc_id = int(doc_features["doc_id"])
        self.doc_source = doc_features["source"]

        ## sentences
        documents_sentence = json_dict["sentences"]
        self.sentences = []
        for sentence in documents_sentence:
            if language == "cn":
                self.sentences.append([x.encode('utf-8') for x in sentence])
            else:
                self.sentences.append(sentence)

        ## build mentions
        mention_dict = json_dict["mentions"]
        self.mentions = [None]*len(mention_dict.keys())
        for mention_num in mention_dict.keys():
            mention = Mention(mention_dict[mention_num]) 
            self.mentions[mention.mention_num] = mention

        ## build pair feature
        pair_feature_dict = json_dict["pair_features"]
        self.pair_feature = {}
        for pairs in pair_feature_dict.keys():
            self.pair_feature[(int(pairs.split(" ")[0]),int(pairs.split(" ")[1]))]=numpy.array(pair_feature_dict[pairs])

        ## gold_chain
        self.gold_chain = gold_chain
        self.gold_chain_dict = {}
        for chian in self.gold_chain:
            for mention_item in chain:
                self.gold_chain_dict[mention_item] = chain

    def is_coreference(self,ment1,ment2):
        if ment1.mention_id in self.gold_chain_dict:
            if ment2.mention_id in self.gold_chain_dict:
                if self.gold_chain_dict[ment1.mention_id] == self.gold_chain_dict[ment2.mention_id]:
                    return True
        return False

def main():

    f = open(args.dev_data)
    gold_file = open(args.dev_gold)
    while True:
        line = f.readline()
        if not line:break
        line = line.strip()
        s = json.loads(line)

        line = gold_file.readline()
        js = json.loads(line)
        gold_chain = js[js.keys()[0]]

        document = Document(s,gold_chain)

if __name__ == "__main__":
    main()
