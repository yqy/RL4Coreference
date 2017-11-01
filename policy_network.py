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

print >> sys.stderr, os.getpid()

random.seed(args.random_seed)

def sample_action(action_probability):
    action = numpy.random.choice(numpy.arange(len(action_probability)),p=action_probability)
    return action
def choose_action(action_probability):
    ac_list = list(action_probability)
    action = ac_list.index(max(ac_list))
    return action

def get_reward(cluster_info,gold_info,max_cluster_num):
    ev_document = get_evaluation_document(cluster_info,gold_info,max_cluster_num)
    p,r,f = evaluation.evaluate_documents([ev_document],evaluation.b_cubed)
    #p,r,f = evaluation.evaluate_documents([ev_document],evaluation.muc)
    #print >> sys.stderr, p,r,f
    return f

def get_evaluation_document(cluster_info,gold_info,max_cluster_num):
    predict = []
    for i in range(max_cluster_num):
        predict.append([])
    for mention_num in range(len(cluster_info)):
        cluster_num = cluster_info[mention_num]
        predict[cluster_num].append(mention_num)
    #print >> sys.stderr," * Gold:",gold_info
    #print >> sys.stderr," * Predict:",predict
    ev_document = evaluation.EvaluationDocument(gold_info,predict)
    return ev_document


def batch_generater(train_case, max_batch_size = 64):

    total_num = len(train_case)
    batch_num = (total_num/max_batch_size)+1

    for i in range(batch_num):
        start = i*max_batch_size
        end = (i+1)*max_batch_size
        this_train_batch = train_case[start:end]

        train_batch_list = []
        mask_batch_list = []

        if len(this_train_batch) <= 1:
            continue
        max_length = len(list(this_train_batch[-1]))

        for i in range(len(this_train_batch)):
            this_train_cas = list(this_train_batch[i])
            add_zeros = [[0.0]*(1374 if args.language=="en" else 1738)]
            train_case_in_batch = this_train_cas + (max_length - len(this_train_cas))*add_zeros
            mask_in_batch = [1]*len(this_train_cas) + [0]*(max_length - len(this_train_cas))

            mask_batch_list.append(mask_in_batch)
            train_batch_list.append(train_case_in_batch)

        yield numpy.array(train_batch_list),numpy.array(mask_batch_list)

def generate_input_case(doc_mention_arrays,doc_pair_arrays):

    train_case = []

    mentions_num = len(doc_mention_arrays)

    for i in range(mentions_num):
        mention_array = doc_mention_arrays[i]
        this_train_case = []

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

        this_train_case = numpy.array(this_train_case)

        train_case.append(this_train_case)

    return train_case

def generate_policy_case(doc_mention_arrays,doc_pair_arrays,gold_chain=[],network=None):
    reward = 0.0

    cluster_info = []
    new_cluster_num = 0

    train_case = generate_input_case(doc_mention_arrays,doc_pair_arrays)

    action_list = []

    train_list = []
    mask_list = []
    for train_batch_list, mask_batch_list in batch_generater(train_case):

        train_list.append(train_batch_list)
        mask_list.append(mask_batch_list)

        action_probabilities = list(network.predict_batch(train_batch_list,mask_batch_list)[0])

        actions = []
        for action_probability in action_probabilities:
            action = sample_action(action_probability)
            actions.append(action)

            if (action-1) == -1: # -1 means a new cluster
                should_cluster = new_cluster_num
                new_cluster_num += 1
            else:
                should_cluster = cluster_info[action-1]

            cluster_info.append(should_cluster)

        action_list.append(actions)

    reward = get_reward(cluster_info,gold_chain,new_cluster_num)
    reward_list = []
    for i in range(len(train_list)):
        reward_list.append([reward]*len(train_list[i]))

    return train_list,mask_list,action_list,reward_list

def generate_policy_test(doc_mention_arrays,doc_pair_arrays,gold_chain=[],network=None):
    cluster_info = []
    new_cluster_num = 0

    train_case = generate_input_case(doc_mention_arrays,doc_pair_arrays)

    for train_batch_list, mask_batch_list in batch_generater(train_case):

        action_probabilities = list(network.predict_batch(train_batch_list,mask_batch_list)[0])

        for action_probability in action_probabilities:

            action = choose_action(action_probability)

            if (action-1) == -1: # -1 means a new cluster
                should_cluster = new_cluster_num
                new_cluster_num += 1
            else:
                should_cluster = cluster_info[action-1]

            cluster_info.append(should_cluster)

    ev_document = get_evaluation_document(cluster_info,gold_chain,new_cluster_num)

    return ev_document
