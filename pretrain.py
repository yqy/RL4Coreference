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

def batch_generater_pretrain(train_case, gold_dict, max_batch_size = 64):

    total_num = len(train_case)
    batch_num = (total_num/max_batch_size)+1

    numpy.random.shuffle(train_case)

    for i in range(batch_num):
        start = i*max_batch_size
        end = (i+1)*max_batch_size
        this_train_batch = train_case[start:end]

        max_length = 0

        this_lable_batch = []
        for index in range(len(this_train_batch)):
            length_of_this_train_case = len(this_train_batch[index])
            if length_of_this_train_case >= max_length:
                max_length = length_of_this_train_case
            lables = [0]*length_of_this_train_case
            index_in_chain = length_of_this_train_case
            #if this_train_batch[index] has n candidates, means that this_train_batch[index] is the index-n mention (begin with 0) in mention list.
            if gold_dict.has_key(index_in_chain):
                for j in gold_dict[index_in_chain]:
                    if j < index_in_chain and j >= 0:
                        lables[j] = 1
            this_lable_batch.append(lables)

        train_batch_list = []
        mask_batch_list = []
        lable_batch_list = []

        if len(this_train_batch) <= 1:
            continue

        neg_num = 0
        pos_num = 0
        for i in range(len(this_train_batch)):
            '''
            add = True
            if this_lable[0] == 1: ## neg
                if neg_num >= pos_num:
                    ra = random.randint(0,neg_num-pos_num)
                    if ra == 0:
                        add = True
                        neg_num += 1
                    else:
                        add = False
            else:
                pos_num += 1

            if not add:
                continue 
            '''
            ## if there is no coreference chain, continue
            if sum(this_lable_batch[i]) == 0:
                continue

            this_lable = list(this_lable_batch[i])

            this_train_cas = list(this_train_batch[i])
            add_zeros = [[0.0]*(1374 if args.language=="en" else 1738)]
            train_case_in_batch = this_train_cas + (max_length - len(this_train_cas))*add_zeros
            train_batch_list.append(train_case_in_batch)

            mask_in_batch = [1]*len(this_train_cas) + [0]*(max_length - len(this_train_cas))
            mask_batch_list.append(mask_in_batch)

            lable_in_batch = this_lable + [0]*(max_length - len(this_train_cas))
            lable_batch_list.append(lable_in_batch)

        if len(lable_batch_list) == 0:
            continue

        yield numpy.array(train_batch_list),numpy.array(mask_batch_list),numpy.array(lable_batch_list)

def generater_pretrain(train_case, gold_dict):

    total_num = len(train_case)

    numpy.random.shuffle(train_case)

    neg_num = 0
    pos_num = 0
    for single_mention_array,this_train in train_case:
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
        if sum(lables) == 0:
            lables[0] = 1

        '''
            
            if neg_num > 0:
                add = False
            else:
                neg_num += 1
        else:
            neg_num -= 1
        '''

        '''
            if neg_num >= pos_num:
                ra = random.randint(0,neg_num-pos_num)
                if ra == 0:
                    add = True
                    neg_num += 1
                else:
                    add = False
        else:
            pos_num += 1
        '''

        if not add:
            continue 

        yield single_mention_array,this_train,lables
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
