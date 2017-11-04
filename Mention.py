#coding=utf8
import json
import sys
import numpy

from conf import *

class Mention():
    # mention id 是在gold里的id对应
    # mention num 是在自己篇章中 按顺序来的
    def __init__(self,mention_dict,language="cn"):
        '''
            dep_relation = 'assmod'
            dep_parent = w
            head_index = 16 

            doc_id = 0
            sent_num = 49
            sentence = [w1,w2,w3...] 
            words = [w3,w4]
            
            mention_id = 80
            mention_num = 344
            mention_type = 'NOMINAL'
            start_index = 16
            end_index = 17
            contained-in-other-mention = 0
        '''


        self.dep_relation = mention_dict["dep_relation"]
        self.sent_num = mention_dict["sent_num"]
        self.head_index = mention_dict["head_index"]
        self.doc_id = mention_dict["doc_id"]
        self.contained_in_other_mention = int(mention_dict["contained-in-other-mention"])
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
        '''
        self.doc_type: int, 0 or 1 dont know its meaning
        self.doc_id: number of document
        self.doc_source: str, example: bc,mz,tc...
        
        self.sentences: list, for each list, is a list of sentence  
            example:[ 
                        [w11,w12,...],
                        [w21,w22,...],
                        [w21,w22,...],
                        ...
                    ]
        self.document_words: list, for each word in the document

        self.mention: See class Mention for more details
        self.pair_feature: dict, (int mention1(mention_num), int mention2(mention_num)) = numpy.array([0,1,0,1..])
            name of each feature: [u'same-speaker', u'antecedent-is-mention-speaker', u'mention-is-antecedent-speaker',
                                   u'relaxed-head-match', u'exact-string-match', u'relaxed-string-match']

        self.gold_chain: list, gold chains, consists of the mention_num (instead of mention_id)
            example: [
                     [1,2,5],
                     [3,7],
                     [4,6]
                     ]

        self.gold_chain_dict: dict, mention_id -> its chain
            exmaple: dict[1] = [1,2,5]
                     dict[2] = [1,2,5]
                     dict[3] = [3,7]
        '''

        ## document_features
        doc_features = json_dict["document_features"]
        self.doc_type = doc_features["type"]
        self.doc_id = int(doc_features["doc_id"])
        self.doc_source = doc_features["source"]

        ## sentences
        documents_sentence = json_dict["sentences"]
        self.sentences = []
        self.document_words = []
        for sentence in documents_sentence:
            if language == "cn":
                self.sentences.append([x.encode('utf-8') for x in sentence])
            else:
                self.sentences.append(sentence)
        for sentence in self.document_words:
            self.document_words += sentence

        ## build mentions
        mention_dict = json_dict["mentions"]
        self.mentions = [None]*len(mention_dict.keys())
        self.id2num = {}
        for mention_num in mention_dict.keys():
            mention = Mention(mention_dict[mention_num]) 
            self.mentions[mention.mention_num] = mention
            self.id2num[mention.mention_id] = mention.mention_num

        ## build pair feature
        pair_feature_dict = json_dict["pair_features"]
        self.pair_feature = {}
        for pairs in pair_feature_dict.keys():
            self.pair_feature[(int(pairs.split(" ")[0]),int(pairs.split(" ")[1]))]=numpy.array(pair_feature_dict[pairs])

        ## gold_chain
        self.gold_chain = self.get_gold_chain(gold_chain)

        self.gold_chain_dict = {}
        for chain in self.gold_chain:
            for mention_item in chain:
                self.gold_chain_dict[mention_item] = chain

    def is_coreference(self,ment1,ment2):
        if ment1.mention_id in self.gold_chain_dict:
            if ment2.mention_id in self.gold_chain_dict:
                if self.gold_chain_dict[ment1.mention_id] == self.gold_chain_dict[ment2.mention_id]:
                    return True
        return False
    def get_gold_chain(self,gold_chain):
        # generate gold chain by using mention_num instead of using mention_id
        rl = []
        neg_num = -1
        for chain in gold_chain:
            this_chain = []
            for mention_id in chain:
                if mention_id in self.id2num:
                    this_chain.append(self.id2num[mention_id])
                else:
                    this_chain.append(neg_num)
                    neg_num -= 1
            rl.append(this_chain)
        return rl

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
        print document.doc_type,document.doc_source

if __name__ == "__main__":
    main()
