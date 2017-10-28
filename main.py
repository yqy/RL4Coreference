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

import cPickle
sys.setrecursionlimit(1000000)

print >> sys.stderr, os.getpid()

random.seed(args.random_seed)

def choose_action(action_probability):
    return 0

def get_reward(cluster_info,gold_info,max_cluster_num):
    predict = [[]]*max_cluster_num
    for mention_num in cluster_info.keys():
        cluster_num = cluster_info[mention_num]
        predict[cluster_num].append(mention_num)
    #print gold_info,predict
    return 0

def generate_policy_case(doc_mention_arrays,doc_pair_arrays,gold_chain=[],network=None):
    train_case = []
    action_case = []
    reward = 0.0

    cluster_info = {}
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
            pair_features = doc_pair_arrays[j*mentions_num + i] 
            this_input = numpy.append(mention_array,mention_in_cluster_array)
            this_input = numpy.append(this_input,pair_features) 
            this_train_case.append(this_input)

        this_train_case = numpy.array(this_train_case)

        #action_probability = network.predict(this_train_case)
        action_probability = 1

        action = choose_action(action_probability)
        action_case.append(action)

        if (action-1) in cluster_info:
            should_cluster = cluster_info[action-1]
        else:
            should_cluster = new_cluster_num
            new_cluster_num += 1

        cluster_info[i] = should_cluster
    #print cluster_info

    reward = get_reward(cluster_info,gold_chain,new_cluster_num)
    
def main():

    embedding_dir = args.embedding+args.language
    print >> sys.stderr,"Read Embedding from %s ..."%embedding_dir
    embedding_dimention = 50
    if args.language == "cn":
        embedding_dimention = 64
    w2v = word2vec.Word2Vec(embedding_dir,embedding_dimention)

    train_docs,dev_docs,test_docs = DataGenerate.get_doc_data()

    # for mention_array: list
    # for mention_pair_array: dict
    train_doc_mention_arrays,train_doc_pair_arrays = DataGenerate.get_arrays(train_docs,"train",w2v)
    print train_doc_mention_arrays.shape
    test_doc_mention_arrays,test_doc_pair_arrays = DataGenerate.get_arrays(test_docs,"test",w2v)
    print test_doc_mention_arrays.shape
    dev_doc_mention_arrays,dev_doc_pair_arrays = DataGenerate.get_arrays(dev_docs,"dev",w2v)

    train_docs = None
    dev_docs = None
    test_docs = None

    for i in range(len(train_doc_mention_arrays)):
        generate_policy_case(train_doc_mention_arrays[i],train_doc_pair_arrays[i])

if __name__ == "__main__":
    main()
