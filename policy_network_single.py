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

def get_reward_trick(cluster_info,gold_dict,max_cluster_num):
    this_cluster = cluster_info[-1]
    this_index = len(cluster_info)-1
    if this_index in gold_dict:
        if this_cluster == (max_cluster_num-1): # it is a new cluster
            reward = 1.0
            for ids in gold_dict[this_index]:
                if ids < this_index:
                    reward = 0.0
                    break
        else:
            al = 0
            right = 0
            for i in range(len(cluster_info)-1):
                if cluster_info[i] == this_cluster:
                    al += 1
                    if i in gold_dict[this_index]:
                        right += 1
            reward = float(right)/float(al)
    else: # it should be a new cluster
        if this_cluster == (max_cluster_num-1):
            reward = 1.0
        else:
            reward = 0.0
    return reward*0.5

def get_evaluation_document(cluster_info,gold_info,max_cluster_num):
    predict = []
    predict_dict = {}
    #for i in range(max_cluster_num):
    #    predict.append([])
    for mention_num in range(len(cluster_info)):
        cluster_num = cluster_info[mention_num]
        predict_dict.setdefault(cluster_num,[])
        predict_dict[cluster_num].append(mention_num)
        #predict[cluster_num].append(mention_num)
    for k in sorted(predict_dict.keys()):
        predict.append(predict_dict[k])
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
    reward_list = []
    max_cluster_at_t = []

    gold_dict = {}
    for cs in gold_chain:
        for nid in cs:
            gold_dict[nid] = cs

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
        max_cluster_at_t.append(new_cluster_num)

        #this_reward = get_reward_trick(cluster_info,gold_dict,new_cluster_num)
        #print this_reward
        #reward_list.append(this_reward)

    for i in range(len(cluster_info)):
        this_cluster = cluster_info[i]
        average_fake_reward = []
        for nc in range(max_cluster_at_t[i]):
            new_cluster_info = cluster_info[:i] + [nc] + cluster_info[i+1:]
            #print this_cluster
            #print new_cluster_info
            average_fake_reward.append(get_reward(new_cluster_info,gold_chain,new_cluster_num))
        average_reward = 0.0 if len(average_fake_reward) == 0 else sum(average_fake_reward)/float(len(average_fake_reward))

        reward_list.append(average_reward)

    reward = get_reward(cluster_info,gold_chain,new_cluster_num)

    indexs = range(len(actions))

    numpy.random.shuffle(indexs) ## for the first mention, it has no mention-pair information, thus should not trained

    for i in indexs:
        single,tc = train_case[i]
        if len(tc) == 0:
            continue
        #yield single, tc, actions[i], reward, action_p[i]
        yield single, tc, actions[i], reward-reward_list[i], action_p[i]

    #for train_batch_list, mask_batch_list, action_batch_list in items_in_batch:
    #    yield train_batch_list, mask_batch_list, action_batch_list, [reward]*len(train_batch_list)


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
    for single,tc in train_case:
        if len(tc) == 0:
            action_probability = numpy.array([1])
        else:
            action_probability = network.predict(single,tc)[0]
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

    indexs = range(len(actions))

    numpy.random.shuffle(indexs) ## for the first mention, it has no mention-pair information, thus should not trained

    for i in indexs:
        single,tc = train_case[i]
        if len(tc) == 0:
            continue
        yield single, tc, actions[i], reward

    #for train_batch_list, mask_batch_list, action_batch_list in items_in_batch:
    #    yield train_batch_list, mask_batch_list, action_batch_list, [reward]*len(train_batch_list)

#def generate_policy_test(doc_mention_arrays,doc_pair_arrays,gold_chain=[],network=None):
def generate_policy_test(train_case,gold_chain=[],network=None):
    cluster_info = []
    new_cluster_num = 0

    #train_case = generate_input_case(doc_mention_arrays,doc_pair_arrays)

    for single,tc in train_case:
        if len(tc) == 0:
            action_probability = [1]
        else:
            action_probability = list(network.predict(single,tc)[0])
        action = choose_action(action_probability)

        if (action-1) == -1: # -1 means a new cluster
            should_cluster = new_cluster_num
            new_cluster_num += 1
        else:
            should_cluster = cluster_info[action-1]

        cluster_info.append(should_cluster)

    ev_document = get_evaluation_document(cluster_info,gold_chain,new_cluster_num)

    return ev_document

