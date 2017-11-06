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
import evaluation

import cPickle
sys.setrecursionlimit(1000000)

random.seed(args.random_seed)

def sample_action(action_probability):
    ac = action_probability/sum(action_probability)
    action = numpy.random.choice(numpy.arange(len(ac)),p=ac)
    return action
def choose_action(action_probability):
    ac_list = list(action_probability)
    action = ac_list.index(max(ac_list))
    return action

def get_reward(cluster_info,gold_info,max_cluster_num):
    ev_document = get_evaluation_document(cluster_info,gold_info,max_cluster_num)
    p,r,f = evaluation.evaluate_documents([ev_document],evaluation.b_cubed)
    #print >> sys.stderr, p,r,f
    return f

def get_evaluation_document(cluster_info,gold_info,max_cluster_num):
    predict = []
    for i in range(max_cluster_num):
        predict.append([])
    for mention_num in range(len(cluster_info)):
        cluster_num = cluster_info[mention_num]
        predict[cluster_num].append(mention_num)
    ev_document = evaluation.EvaluationDocument(gold_info,predict)
    return ev_document


def generate_input_case(doc_mention_arrays,doc_pair_arrays,pretrain=False):

    train_case = []

    mentions_num = len(doc_mention_arrays)

    for i in range(mentions_num):
        mention_array = doc_mention_arrays[i]
        this_train_case = []

        if not pretrain:
            ## add a Noun cluster
            Noun_cluster_array = numpy.array([0.0]*len(mention_array))
            this_input = numpy.append(mention_array,Noun_cluster_array)
            this_input = numpy.append(this_input,numpy.array([0.0]*28))
            this_train_case.append(this_input)

        for j in range(0,i):
            mention_in_cluster_array = doc_mention_arrays[j]
            pair_features = doc_pair_arrays[(2*mentions_num-j-1)*j/2 + i-j -1]  #等差数列算出
            this_input = numpy.append(mention_array,mention_in_cluster_array)
            this_input = numpy.append(this_input,pair_features) 
            this_train_case.append(this_input)

        this_train_case = numpy.array(this_train_case,dtype = numpy.float32)

        train_case.append(this_train_case)

    return train_case

def generate_policy_case(doc_mention_arrays,doc_pair_arrays,gold_chain=[],network=None):
    reward = 0.0

    train_case = generate_input_case(doc_mention_arrays,doc_pair_arrays)

    actions_dict = {}

    items_in_batch = []

    st = timeit.default_timer()

    start_time = timeit.default_timer()
    #print len(train_case)
    
    cluster_info = []
    new_cluster_num = 0
    actions = []
    for tc in train_case:
        action_probability = list(network.predict(tc)[0])
        action = sample_action(action_probability)
        actions.append(action)

        if (action-1) == -1: # -1 means a new cluster
            should_cluster = new_cluster_num
            new_cluster_num += 1
        else:
            should_cluster = cluster_info[action-1]

        cluster_info.append(should_cluster)

    reward = get_reward(cluster_info,gold_chain,new_cluster_num)

    indexs = range(len(actions))
    numpy.random.shuffle(indexs)

    for i in indexs:
        yield train_case[i],actions[i],reward

    #for train_batch_list, mask_batch_list, action_batch_list in items_in_batch:
    #    yield train_batch_list, mask_batch_list, action_batch_list, [reward]*len(train_batch_list)

def generate_policy_test(doc_mention_arrays,doc_pair_arrays,gold_chain=[],network=None):
    cluster_info = []
    new_cluster_num = 0

    train_case = generate_input_case(doc_mention_arrays,doc_pair_arrays)

    for tc in train_case:
        action_probability = list(network.predict(tc)[0])
        action = sample_action(action_probability)
        actions.append(action)

        if (action-1) == -1: # -1 means a new cluster
            should_cluster = new_cluster_num
            new_cluster_num += 1
        else:
            should_cluster = cluster_info[action-1]

        cluster_info.append(should_cluster)

    ev_document = get_evaluation_document(cluster_info,gold_chain,new_cluster_num)

    return ev_document

