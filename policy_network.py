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
    action = action_probability.index(max(action_probability))
    return action


def get_reward(cluster_info,gold_info,max_cluster_num):
    ev_document = get_evaluation_document(cluster_info,gold_info,max_cluster_num)
    p,r,f = evaluation.evaluate_documents([ev_document],evaluation.b_cubed)
    print >> sys.stderr, p,r,f
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


def batch_generate(train_case, max_batch_size = 256):

    train_list = []
    mask_list = []
    
    total_num = len(train_case)
    batch_num = (total_num/max_batch_size)+1

    for i in range(batch_num):
        start = i*max_batch_size
        end = (i+1)*max_batch_size
        this_train_batch = train_case[start:end]

        train_batch_list = []
        mask_batch_list = []

        max_length = len(list(this_train_batch[-1]))

        for i in range(len(this_train_batch)):
            this_train_cas = list(this_train_batch[i])
            add_zeros = [[0.0]*(1374 if args.language=="en" else 1738)]
            train_case_in_batch = this_train_cas + (max_length - len(this_train_cas))*add_zeros
            mask_in_batch = [1]*len(this_train_cas) + [0]*(max_length - len(this_train_cas))

            mask_batch_list.append(mask_in_batch)
            train_batch_list.append(train_case_in_batch)

        train_list.append(train_batch_list)
        mask_list.append(mask_batch_list)

    train_list = train_list
    mask_list = mask_list

    return train_list,mask_list

def generate_policy_case(doc_mention_arrays,doc_pair_arrays,gold_chain=[],network=None):
    train_case = []
    action_case = []
    reward = 0.0

    cluster_info = []
    new_cluster_num = 0

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
            #pair_features = doc_pair_arrays[(j,i)] 
            pair_features = doc_pair_arrays[(2*mentions_num-j-1)*j/2 + i-j -1]  #等差数列算出
            this_input = numpy.append(mention_array,mention_in_cluster_array)
            this_input = numpy.append(this_input,pair_features) 
            this_train_case.append(this_input)


        this_train_case = numpy.array(this_train_case)

        train_case.append(this_train_case)

    train_list,mask_list = batch_generate(train_case) 

    for i in range(len(train_list)):
        train_batch_list = train_list[i]
        mask_batch_list = mask_list[i]
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
        # cluster_info: save the cluster information for each mention

        action_case.append(actions)

    reward = get_reward(cluster_info,gold_chain,new_cluster_num)
    reward_list = []
    for i in range(len(train_list)):
        reward_list.append([reward]*len(train_list[i]))

    return train_list,mask_list,action_case,reward_list

def generate_policy_case_with_train_mask(train_list,mask_list,gold_chain=[],network=None):
    action_case = []
    reward = 0.0

    cluster_info = []
    new_cluster_num = 0

    mentions_num = len(train_list)

    action_probabilities = list(network.predict_batch(train_list,mask_list)[0])
    for action_probability in action_probabilities:
        action = sample_action(action_probability)
        action_case.append(action)

        if (action-1) == -1: # -1 means a new cluster
            should_cluster = new_cluster_num
            new_cluster_num += 1
        else:
            should_cluster = cluster_info[action-1]

        cluster_info.append(should_cluster)
        # cluster_info: save the cluster information for each mention

    reward = get_reward(cluster_info,gold_chain,new_cluster_num)
    reward_list = [reward]*mentions_num

    return action_case,reward_list

def generate_policy_test(doc_mention_arrays,doc_pair_arrays,gold_chain=[],network=None):
    train_case = []
    action_case = []

    cluster_info = []
    new_cluster_num = 0

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
            #pair_features = doc_pair_arrays[(j,i)] 
            pair_features = doc_pair_arrays[(2*mentions_num-j-1)*j/2 + i-j -1]  #等差数列算出
            this_input = numpy.append(mention_array,mention_in_cluster_array)
            this_input = numpy.append(this_input,pair_features) 
            this_train_case.append(this_input)


        this_train_case = numpy.array(this_train_case)

        train_case.append(this_train_case)

    train_list,mask_list = batch_generate(train_case) 
    for i in range(len(train_list)):
        train_batch_list = train_list[i]
        mask_batch_list = mask_list[i]

        action_probabilities = list(network.predict_batch(train_batch_list,mask_batch_list)[0])

        for action_probability in action_probabilities:
            action = choose_action(action_probability)
            action_case.append(action)

            if (action-1) == -1: # -1 means a new cluster
                should_cluster = new_cluster_num
                new_cluster_num += 1
            else:
                should_cluster = cluster_info[action-1]

            cluster_info.append(should_cluster)

    ev_document = get_evaluation_document(cluster_info,gold_chain,new_cluster_num)

    return ev_document,train_list,mask_list

def generate_policy_test_with_train_mask(train_list,mask_list,gold_chain=[],network=None):

    action_case = []
    cluster_info = []
    new_cluster_num = 0

    mentions_num = len(train_list)

    action_probabilities = list(network.predict_batch(train_list,mask_list)[0])
    for action_probability in action_probabilities:
        action = sample_action(action_probability)
        action_case.append(action)

        if (action-1) == -1: # -1 means a new cluster
            should_cluster = new_cluster_num
            new_cluster_num += 1
        else:
            should_cluster = cluster_info[action-1]

        cluster_info.append(should_cluster)

    ev_document = get_evaluation_document(cluster_info,gold_chain,new_cluster_num)

    return ev_document


def main():

    embedding_dir = args.embedding+args.language
    print >> sys.stderr,"Read Embedding from %s ..."%embedding_dir
    embedding_dimention = 50
    if args.language == "cn":
        embedding_dimention = 64
    w2v = word2vec.Word2Vec(embedding_dir,embedding_dimention)

    print >> sys.stderr,"Building Model ..."

    #network_model
    if os.path.isfile("./model/network_model."+args.language):
        read_f = file('./model/network_model.'+args.language, 'rb')
        network_model = cPickle.load(read_f)
        print >> sys.stderr,"Read model from ./model/network_model."+args.language
    else:
        inpt_dimention = 1738
        if args.language == "en":
            inpt_dimention = 1374
        network_model = network.NetWork(inpt_dimention,1000)
        print >> sys.stderr,"save model ..."
        save_f = file('./model/network_model.'+args.language, 'wb')
        cPickle.dump(network_model, save_f, protocol=cPickle.HIGHEST_PROTOCOL)
        save_f.close()

    train_docs = DataGenerate.doc_data_generater("train")
    dev_docs = DataGenerate.doc_data_generater("dev")
    test_docs = DataGenerate.doc_data_generater("test")

    # for mention_array: list
    # for mention_pair_array: list

    for train_doc_mention_array,train_doc_pair_array,train_doc_gold_chain in DataGenerate.array_generater(train_docs,"train",w2v):
        #generate_policy_case(train_doc_mention_array,train_doc_pair_array,train_doc_gold_chain)
        train_case,action_case,reward = generate_policy_case(train_doc_mention_array,train_doc_pair_array,train_doc_gold_chain,network_model)
        

    #for dev_doc_mention_array,dev_doc_pair_array,dev_doc_gold_chain in DataGenerate.array_generater(dev_docs,"dev",w2v):
    #    generate_policy_case(dev_doc_mention_array,dev_doc_pair_array,dev_doc_gold_chain)
    #for test_doc_mention_array,test_doc_pair_array,test_doc_gold_chain in DataGenerate.array_generater(test_docs,"test",w2v):
    #    generate_policy_case(test_doc_mention_array,test_doc_pair_array,test_doc_gold_chain)

if __name__ == "__main__":
    main()
