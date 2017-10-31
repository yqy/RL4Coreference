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
    
    for echo in range(20):
        print "ECHO:",echo
        for train_doc_mention_array,train_doc_pair_array,train_doc_gold_chain in DataGenerate.array_generater(train_docs,"train",w2v):
            train_list,mask_list,action_case,reward_list = policy_network.generate_policy_case(train_doc_mention_array,train_doc_pair_array,train_doc_gold_chain,network_model)
            for batch_num in len(train_list):
                train_batch = train_list[i]
                mask_batch = mask_list[i]
                action_batch = action_case[i]
                reward_batch = reward_list[i]
            network_model.train_step(train_batch,mask_batch,action_batch,reward_batch,0.01)
    
        dev_docs = []
        for dev_doc_mention_array,dev_doc_pair_array,dev_doc_gold_chain in DataGenerate.array_generater(dev_docs,"dev",w2v):
            ev_doc,train_l,ml = policy_network.generate_policy_test(dev_doc_mention_array,dev_doc_pair_array,dev_doc_gold_chain,network_model)
            dev_docs.append(ev_doc)
        print "DEV"
        mp,mr,mf = evaluation.evaluate_documents(dev_docs,evaluation.muc)
        print "MUC: recall: %f precision: %f  f1: %f"%(mr,mp,mf)
        bp,br,bf = evaluation.evaluate_documents(dev_docs,evaluation.b_cubed)
        print "BCUBED: recall: %f precision: %f  f1: %f"%(br,bp,bf)
        cp,cr,cf = evaluation.evaluate_documents(dev_docs,evaluation.ceafe)
        print "CEAF: recall: %f precision: %f  f1: %f"%(cr,cp,cf)
    
        test_docs = []
        for test_doc_mention_array,test_doc_pair_array,test_doc_gold_chain in DataGenerate.array_generater(test_docs,"test",w2v):
            ev_doc,train_l,ml = policy_network.generate_policy_test(test_doc_mention_array,test_doc_pair_array,test_doc_gold_chain,network_model)
            test_docs.append(ev_doc)
        print "TEST"
        mp,mr,mf = evaluation.evaluate_documents(test_docs,evaluation.muc)
        print "MUC: recall: %f precision: %f  f1: %f"%(mr,mp,mf)
        bp,br,bf = evaluation.evaluate_documents(test_docs,evaluation.b_cubed)
        print "BCUBED: recall: %f precision: %f  f1: %f"%(br,bp,bf)
        cp,cr,cf = evaluation.evaluate_documents(test_docs,evaluation.ceafe)
        print "CEAF: recall: %f precision: %f  f1: %f"%(cr,cp,cf)



if __name__ == "__main__":
    main()
