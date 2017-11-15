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
    ac = action_probability/action_probability.sum()
    action = numpy.random.choice(numpy.arange(len(ac)),p=ac)
    return action

def sample_action_trick(action_probability,ran_p = 0.05):
    if random.random() <= ran_p:
        action = random.randint(0,len(action_probability)-1)
        return action
    else:
        ac = action_probability/action_probability.sum()
        action = numpy.random.choice(numpy.arange(len(ac)),p=ac)
        return action


def choose_action(action_probability):
    #print action_probability
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
    ev_document = evaluation.EvaluationDocument(gold_info,predict)
    return ev_document


def generate_input_case(doc_mention_arrays,doc_pair_arrays,pretrain=False):

    train_case = []

    mentions_num = len(doc_mention_arrays)

    for i in range(mentions_num):
        mention_array = doc_mention_arrays[i]
        this_train_case = []

        for j in range(0,i):
            mention_in_cluster_array = doc_mention_arrays[j]
            #pair_features = doc_pair_arrays[(2*mentions_num-j-1)*j/2 + i-j -1]  #等差数列算出
            pair_features = doc_pair_arrays[(j,i)]
            this_input = numpy.append(mention_array,mention_in_cluster_array)
            this_input = numpy.append(this_input,pair_features) 
            this_train_case.append(this_input)

        this_train_case = numpy.array(this_train_case,dtype = numpy.float32)

        mention_array_float = numpy.array([mention_array],dtype = numpy.float32)

        train_case.append((mention_array_float,this_train_case))

    return train_case

def generate_policy_case_trick(train_case,gold_chain=[],network=None,ran_p = 0.05):
    reward = 0.0

    #train_case = generate_input_case(doc_mention_arrays,doc_pair_arrays)

    actions_dict = {}

    items_in_batch = []

    st = timeit.default_timer()

    start_time = timeit.default_timer()
    #print len(train_case)
    
    cluster_info = []
    new_cluster_num = 0
    actions = []
    action_p = []
    for single,tc in train_case:
        if len(tc) == 0:
            action_probability = numpy.array([1])
        else:
            action_probability = network.predict(single,tc)[0]
        action = sample_action_trick(action_probability,ran_p)
        action_p.append(action_probability[action])
        #action = choose_action(action_probability)
        actions.append(action)

        if (action-1) == -1: # -1 means a new cluster
            should_cluster = new_cluster_num
            new_cluster_num += 1
        else:
            should_cluster = cluster_info[action-1]

        cluster_info.append(should_cluster)

    reward = get_reward(cluster_info,gold_chain,new_cluster_num)

    indexs = range(len(actions))

    numpy.random.shuffle(indexs) ## for the first mention, it has no mention-pair information, thus should not trained

    for i in indexs:
        single,tc = train_case[i]
        if len(tc) == 0:
            continue
        yield single, tc, actions[i], reward, action_p[i]

    #for train_batch_list, mask_batch_list, action_batch_list in items_in_batch:
    #    yield train_batch_list, mask_batch_list, action_batch_list, [reward]*len(train_batch_list)


def batch_generater(train_case, max_batch_size = 1):

    total_num = len(train_case)

    max_batch_size = 1

    add_zeros = [[0.0]*(1374 if args.language=="en" else 1738)]
    batch_num = (total_num/max_batch_size)+1
    
    for i in range(batch_num):
        start = i*max_batch_size
        end = (i+1)*max_batch_size
        this_train_batch = train_case[start:end]

        train_batch_list = []
        single_batch_list = []
        mask_batch_list = []


        if len(this_train_batch) < 1:
            continue

        sin,tc = this_train_batch[-1]

        max_length = len(list(tc))

        for i in range(len(this_train_batch)):
            this_single_cas, this_train_cas = this_train_batch[i]
            train_case_in_batch = this_train_cas + (max_length - len(this_train_cas))*add_zeros
            mask_in_batch = [1] + [1]*len(this_train_cas) + [0]*(max_length - len(this_train_cas))

            mask_batch_list.append(mask_in_batch)
            single_batch_list.append(this_single_cas)
            train_batch_list.append(train_case_in_batch)

        yield numpy.array(train_batch_list,dtype = numpy.float32),numpy.array(single_batch_list,dtype = numpy.float32),numpy.array(mask_batch_list,dtype = numpy.int8)

def batch_generater_shuffle(train_case, actions, max_batch_size = 128):

    total_num = len(train_case)

    if total_num >= 500 and total_num <= 1000:
        max_batch_size = 64
    elif total_num > 1000:
        max_batch_size = 32

    add_zeros = [[0.0]*(1374 if args.language=="en" else 1738)]
    batch_num = (total_num/max_batch_size)+1

    index_list = range(total_num)
    numpy.random.shuffle(index_list)
    
    for i in range(batch_num):
        start = i*max_batch_size
        end = (i+1)*max_batch_size
        this_index_batch = index_list[start:end]

        if len(this_index_batch) < 1:
            continue

        this_train_batch = train_case[start:end]

        train_batch_list = []
        single_batch_list = []
        mask_batch_list = []
        action_batch_list = []

        max_length = max([len(train_case[x][1]) for x in this_index_batch])

        for i in this_train_batch:

            this_single_cas, this_train_cas = this_train_batch[i]
            train_case_in_batch = this_train_cas + (max_length - len(this_train_cas))*add_zeros
            mask_in_batch = [1] + [1]*len(this_train_cas) + [0]*(max_length - len(this_train_cas))

            mask_batch_list.append(mask_in_batch)
            single_batch_list.append(this_single_cas)
            train_batch_list.append(train_case_in_batch)
            action_batch_list.append(actions[i])

        yield numpy.array(train_batch_list,dtype = numpy.float32),numpy.array(single_batch_list,dtype = numpy.float32),numpy.array(mask_batch_list,dtype = numpy.int8),numpy.array(action_batch_list,dtype = numpy.int8)


#def generate_policy_case(doc_mention_arrays,doc_pair_arrays,gold_chain=[],network=None):
def generate_policy_case(train_case,gold_chain=[],network=None):
    reward = 0.0

    #train_case = generate_input_case(doc_mention_arrays,doc_pair_arrays)

    actions_dict = {}

    items_in_batch = []

    st = timeit.default_timer()

    start_time = timeit.default_timer()
    #print len(train_case)
    
    cluster_info = []
    new_cluster_num = 0
    actions = []
    for tc,single,mask in batch_generater(train_case): ## only deal with one case each time, it is a fake batch
        if len(tc[0]) == 0:
            action_probability = numpy.array([1])
        else:
            action_probability = network.predict(single,tc,mask)[0][0]

        action = sample_action(action_probability)
        #action = choose_action(action_probability)
        actions.append(action)

        if action == 0: # 0 means a new cluster
            should_cluster = new_cluster_num
            new_cluster_num += 1
        else:
            should_cluster = cluster_info[action-1]

        cluster_info.append(should_cluster)

    reward = get_reward(cluster_info,gold_chain,new_cluster_num)

    #for i in indexs:
    #    single,tc = train_case[i]
    #    if len(tc) == 0:
    #        continue
    #    yield single, tc, actions[i], reward

    for train_batch_list, single_batch_list, mask_batch_list, action_batch_list in batch_generater_shuffle(train_case,actions):
        yield train_batch_list, single_batch_list, mask_batch_list, action_batch_list, reward

def generate_policy_test(train_case,gold_chain=[],network=None):
    cluster_info = []
    new_cluster_num = 0

    #train_case = generate_input_case(doc_mention_arrays,doc_pair_arrays)

    for tc,single,mask in batch_generater(train_case): ## only deal with one case each time, it is a fake batch
    #for single,tc in train_case:
        if len(tc[0]) == 0:
            action_probability = [1]
        else:
            action_probability = list(network.predict(single,tc,mask)[0][0])
        action = choose_action(action_probability)

        if (action-1) == -1: # -1 means a new cluster
            should_cluster = new_cluster_num
            new_cluster_num += 1
        else:
            should_cluster = cluster_info[action-1]

        cluster_info.append(should_cluster)

    ev_document = get_evaluation_document(cluster_info,gold_chain,new_cluster_num)

    return ev_document

