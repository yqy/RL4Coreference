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
import policy_network
import network
import network_batch as network

import cPickle
sys.setrecursionlimit(1000000)

print >> sys.stderr, os.getpid()

random.seed(args.random_seed)

def batch_generate(train_case,action_case,reward):
    reward_list = []
    train_list = []
    action_list = []
    mask_list = []

    max_length = len(list(train_case[:-1]))

    for i in range(len(train_case)):
        this_train_cas = list(train_case[i])
        add_zeros = [[0.0]*(1374 if args.language=="en" else 1738)]
        train_case_in_batch = this_train_cas + (max_length - len(this_train_cas))*add_zeros
        mask_in_batch = [1]*len(this_train_cas) + [0]*(max_length - len(this_train_cas))
        
        mask_list.append(mask_in_batch)
        train_list.append(train_case_in_batch)
        action_list.append(action_case[i])
        reward_list.append(reward)

    reward_list = numpy.array(reward_list)
    train_list = numpy.array(train_list)
    action_list = numpy.array(action_list)
    mask_list = numpy.array(mask_list)

    print train_list
    print train_list.shape

    return train_list,mask_list,action_list,reward_list

def main():

    embedding_dir = args.embedding+args.language
    print >> sys.stderr,"Read Embedding from %s ..."%embedding_dir
    embedding_dimention = 50
    if args.language == "cn":
        embedding_dimention = 64
    w2v = word2vec.Word2Vec(embedding_dir,embedding_dimention)

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
        train_case,action_case,reward = policy_network.generate_policy_case(train_doc_mention_array,train_doc_pair_array,train_doc_gold_chain,network_model)
        train_list,mask_list,action_list,reward_list = batch_generate(train_case,action_case,reward)

        network_model.train_step(train_list,mask_list,action_list,reward_list,0.5)

    for train_doc_mention_array,train_doc_pair_array,train_doc_gold_chain in DataGenerate.array_generater(train_docs,"train",w2v):
        policy_network.generate_policy_case(train_doc_mention_array,train_doc_pair_array,train_doc_gold_chain,network_model)


#    for dev_doc_mention_array,dev_doc_pair_array,dev_doc_gold_chain in DataGenerate.array_generater(dev_docs,"dev",w2v):
#        policy_network.generate_policy_case(dev_doc_mention_array,dev_doc_pair_array,dev_doc_gold_chain)
#    for test_doc_mention_array,test_doc_pair_array,test_doc_gold_chain in DataGenerate.array_generater(test_docs,"test",w2v):
#        policy_network.generate_policy_case(test_doc_mention_array,test_doc_pair_array,test_doc_gold_chain)

if __name__ == "__main__":
    main()
