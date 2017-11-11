#coding=utf8

import sys
import os
import json
import random
import numpy
import timeit

from conf import *

import Mention
import Reader
import word2vec
import DataGenerate
#import policy_network
import policy_network_single as policy_network

import cPickle
sys.setrecursionlimit(1000000)

random.seed(args.random_seed)

def generater_pretrain(train_case, gold_dict):

    total_num = len(train_case)

    numpy.random.shuffle(train_case)

    neg_num = 0
    pos_num = 0
    for single_mention_array,single_index,this_train,this_train_index in train_case:
        if len(this_train) == 0:
            continue
        length_of_this_train_case = len(this_train)
        index_in_chain = length_of_this_train_case
        lables = [0]*(length_of_this_train_case+1)

        if gold_dict.has_key(index_in_chain):
            for j in gold_dict[index_in_chain]:
                if (j+1) < index_in_chain and j >= 0:
                    lables[j+1] = 1
                
        add = True

        yield single_mention_array,single_index,this_train,this_train_index,lables
        #numpy.array(train_batch_list),numpy.array(mask_batch_list),numpy.array(lable_batch_list)


#def generate_pretrain_case(doc_mention_arrays,doc_pair_arrays,gold_chain=[],network=None):
def generate_pretrain_case(train_case,gold_chain=[],network=None):
    cluster_info = []
    new_cluster_num = 0

    #train_case = policy_network.generate_input_case(doc_mention_arrays,doc_pair_arrays,pretrain=True)
    gold_dict = {}
    lable_in_gold = []
    for chain in gold_chain:
        for item in chain:
            gold_dict[item] = chain

    #return batch_generater_pretrain(train_case[1:],gold_dict)
    return generater_pretrain(train_case,gold_dict)
    ## train_case[0] = Null. because the first mention has no antecedents
