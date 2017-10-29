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

def choose_action(action_probability):
    return 0

def get_reward(cluster_info,gold_info,max_cluster_num):
    predict = [[]]*max_cluster_num
    for mention_num in range(len(cluster_info)):
        cluster_num = cluster_info[mention_num]
        predict[cluster_num].append(mention_num)

    ev_document = evaluation.EvaluationDocument(gold_info,predict)
    p,r,f = evaluation.evaluate_documents([ev_document],evaluation.muc)
    print gold_info
    print predict
    print p,r,f
    #print gold_info,predict
    return 0

def generate_policy_case(doc_mention_arrays,doc_pair_arrays,gold_chain=[],network=None):
    train_case = []
    action_case = []
    reward = 0.0

    cluster_info = []
    new_cluster_num = 0

    mentions_num = len(doc_mention_arrays)

    for i in range(len(doc_mention_arrays)):
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

        #action_probability = network.predict(this_train_case)
        action_probability = 1

        action = choose_action(action_probability)
        action_case.append(action)

        if (action-1) == -1: # -1 means a new cluster
            should_cluster = new_cluster_num
            new_cluster_num += 1
        else:
            should_cluster = cluster_info[action-1]

        cluster_info.append(should_cluster)
        # cluster_info: save the cluster information for each mention

    reward = get_reward(cluster_info,gold_chain,new_cluster_num)
    return train_case,action_case,reward

def generate_cases(mention_a,pair_a,gold_chain):
    for i in range(len(mention_a)):
        yield (mention_a[i],pair_a[i],gold_chain[i])

def main():

    embedding_dir = args.embedding+args.language
    print >> sys.stderr,"Read Embedding from %s ..."%embedding_dir
    embedding_dimention = 50
    if args.language == "cn":
        embedding_dimention = 64
    w2v = word2vec.Word2Vec(embedding_dir,embedding_dimention)

    train_docs,dev_docs,test_docs = DataGenerate.get_doc_data()

    # for mention_array: list
    # for mention_pair_array: list
    
    for train_doc_mention_array,train_doc_pair_array,train_doc_gold_chain in DataGenerate.get_arrays(train_docs,"train",w2v):
        generate_policy_case(train_doc_mention_array,train_doc_pair_array,train_doc_gold_chain)
    #train_doc_mention_arrays,train_doc_pair_arrays,train_doc_gold_chains = DataGenerate.get_arrays(train_docs,"train",w2v)
    #test_doc_mention_arrays,test_doc_pair_arrays,test_doc_gold_chains = DataGenerate.get_arrays(test_docs,"test",w2v)
    #dev_doc_mention_arrays,dev_doc_pair_arrays,dev_doc_gold_chains = DataGenerate.get_arrays(dev_docs,"dev",w2v)

    dev_docs = None
    test_docs = None
    train_docs = None

    #for train_doc_mention_array,train_doc_pair_array,train_doc_gold_chain in generate_cases(train_doc_mention_arrays,train_doc_pair_arrays,train_doc_gold_chains):
        #generate_policy_case(train_doc_mention_array,train_doc_pair_array,train_doc_gold_chain)

if __name__ == "__main__":
    main()
