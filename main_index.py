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
import DataGenerate_index as DataGenerate
import evaluation
#import policy_network
import policy_network_single_index as policy_network
#import network
import network_index as network
import pretrain_index as pretrain

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
    w2v = word2vec.Word2Vec(embedding_dir+".filtered",embedding_dimention)

    #network_model
    if os.path.isfile("./model/network_model_index."+args.language):
        read_f = file('./model/network_model_index.'+args.language, 'rb')
        #read_f = file('./model/network_model_pretrain.'+args.language, 'rb')
        network_model = cPickle.load(read_f)
        print >> sys.stderr,"Read model from ./model/network_model_index."+args.language
    else:
        inpt_dimention = 1738
        single_dimention = 855
        if args.language == "en":
            inpt_dimention = 1374
            single_dimention = 673

        network_model = network.NetWork(inpt_dimention,single_dimention,1000,embedding_dir+".filtered",embedding_dimention)
        print >> sys.stderr,"save model ..."
        save_f = file('./model/network_model_index.'+args.language, 'wb')
        cPickle.dump(network_model, save_f, protocol=cPickle.HIGHEST_PROTOCOL)
        save_f.close()

    train_docs = DataGenerate.doc_data_generater("train")
    dev_docs = DataGenerate.doc_data_generater("dev")
    test_docs = DataGenerate.doc_data_generater("test")

    #pretrain
    times = 0
    for echo in range(20):
        start_time = timeit.default_timer()
        print "Pretrain ECHO:",echo
        cost_this_turn = 0.0
        for cases,gold_chain in DataGenerate.case_generater(train_docs,"train",w2v):
            if len(cases) >= 700:
                continue
            for single_mention_array,single_index,train_list,train_index,lable_list in pretrain.generate_pretrain_case(cases,gold_chain,network_model):
                print network_model.fff(train_list,train_index)
                cost_this_turn += network_model.pre_train_step(single_mention_array,single_index,train_list,train_index,lable_list,0.0001)[0]

        end_time = timeit.default_timer()
        print >> sys.stderr, "PreTrain",echo,"Total cost:",cost_this_turn
        print >> sys.stderr, "PreTRAINING Use %.3f seconds"%(end_time-start_time)

        if echo %4 == 0:
            save_f = file('./model/network_model_pretrain_index.'+args.language, 'wb')
            cPickle.dump(network_model, save_f, protocol=cPickle.HIGHEST_PROTOCOL)
            save_f.close()

    save_f = file('./model/network_model_pretrain_index.'+args.language, 'wb')
    cPickle.dump(network_model, save_f, protocol=cPickle.HIGHEST_PROTOCOL)
    save_f.close()

    print >> sys.stderr,"Begin test on DEV after pertraining"
    
    ## test performance after pretraining
    dev_docs_for_test = []
    num = 0
    for cases,gold_chain in DataGenerate.case_generater(dev_docs,"dev",w2v):
        ev_doc = policy_network.generate_policy_test(cases,gold_chain,network_model)
        dev_docs_for_test.append(ev_doc)
    print "Performance on DEV after PreTRAINING"
    mp,mr,mf = evaluation.evaluate_documents(dev_docs_for_test,evaluation.muc)
    print "MUC: recall: %f precision: %f  f1: %f"%(mr,mp,mf)
    bp,br,bf = evaluation.evaluate_documents(dev_docs_for_test,evaluation.b_cubed)
    print "BCUBED: recall: %f precision: %f  f1: %f"%(br,bp,bf)
    cp,cr,cf = evaluation.evaluate_documents(dev_docs_for_test,evaluation.ceafe)
    print "CEAF: recall: %f precision: %f  f1: %f"%(cr,cp,cf)
    print "##################################################" 
    sys.stdout.flush()
    print >> sys.stderr,"Pre Train done"

    ##train
    train4test = [] # add 5 items for testing the training performance
    add2train = True

    for echo in range(20):
        start_time = timeit.default_timer()
        reward_baseline = []
        cost_this_turn = 0.0

        for cases,gold_chain in DataGenerate.case_generater(train_docs,"train",w2v):
        
            if add2train:
                if random.randint(1,200) == 10:
                    #train4test.append((train_doc_mention_array,train_doc_pair_array,train_doc_gold_chain))
                    train4test.append((cases,gold_chain))
                    if len(train4test) == 5:
                        add2train = False

            this_reward = 0.0

            for single, sindex, train, tindex, action, reward in policy_network.generate_policy_case(cases,gold_chain,network_model):
                cost_this_turn += network_model.train_step(single,sindex,train,tindex,action,reward,0.0001)[0]

        end_time = timeit.default_timer()
        print >> sys.stderr, "Total cost:",cost_this_turn
        print >> sys.stderr, "TRAINING Use %.3f seconds"%(end_time-start_time)

        ## test training performance
        train_docs_for_test = []
        start_time = timeit.default_timer()

        for train_cases,train_doc_gold_chain in train4test:
            ev_doc = policy_network.generate_policy_test(train_cases,train_doc_gold_chain,network_model)
            train_docs_for_test.append(ev_doc)
        print "** Echo: %d **"%echo
        print "TRAIN"
        mp,mr,mf = evaluation.evaluate_documents(train_docs_for_test,evaluation.muc)
        print "MUC: recall: %f precision: %f  f1: %f"%(mr,mp,mf)
        bp,br,bf = evaluation.evaluate_documents(train_docs_for_test,evaluation.b_cubed)
        print "BCUBED: recall: %f precision: %f  f1: %f"%(br,bp,bf)
        cp,cr,cf = evaluation.evaluate_documents(train_docs_for_test,evaluation.ceafe)
        print "CEAF: recall: %f precision: %f  f1: %f"%(cr,cp,cf)
        print

        ## dev
        dev_docs_for_test = []
        start_time = timeit.default_timer()
        #for dev_doc_mention_array,dev_doc_pair_array,dev_doc_gold_chain in DataGenerate.array_generater(dev_docs,"dev",w2v):
            #ev_doc = policy_network.generate_policy_test(dev_doc_mention_array,dev_doc_pair_array,dev_doc_gold_chain,network_model)
        for dev_cases,dev_doc_gold_chain in DataGenerate.case_generater(dev_docs,"dev",w2v):
            ev_doc = policy_network.generate_policy_test(dev_cases,dev_doc_gold_chain,network_model)
            dev_docs_for_test.append(ev_doc)
        print "DEV"
        mp,mr,mf = evaluation.evaluate_documents(dev_docs_for_test,evaluation.muc)
        print "MUC: recall: %f precision: %f  f1: %f"%(mr,mp,mf)
        bp,br,bf = evaluation.evaluate_documents(dev_docs_for_test,evaluation.b_cubed)
        print "BCUBED: recall: %f precision: %f  f1: %f"%(br,bp,bf)
        cp,cr,cf = evaluation.evaluate_documents(dev_docs_for_test,evaluation.ceafe)
        print "CEAF: recall: %f precision: %f  f1: %f"%(cr,cp,cf)
        print 

        end_time = timeit.default_timer()
        print >> sys.stderr, "DEV Use %.3f seconds"%(end_time-start_time)
        sys.stdout.flush()
    
        ## test
        test_docs_for_test = []
        start_time = timeit.default_timer()
        #for test_doc_mention_array,test_doc_pair_array,test_doc_gold_chain in DataGenerate.array_generater(test_docs,"test",w2v):
        for test_cases,test_doc_gold_chain in DataGenerate.case_generater(test_docs,"test",w2v):
            ev_doc = policy_network.generate_policy_test(test_cases,test_doc_gold_chain,network_model)
            test_docs_for_test.append(ev_doc)
        print "TEST"
        mp,mr,mf = evaluation.evaluate_documents(test_docs_for_test,evaluation.muc)
        print "MUC: recall: %f precision: %f  f1: %f"%(mr,mp,mf)
        bp,br,bf = evaluation.evaluate_documents(test_docs_for_test,evaluation.b_cubed)
        print "BCUBED: recall: %f precision: %f  f1: %f"%(br,bp,bf)
        cp,cr,cf = evaluation.evaluate_documents(test_docs_for_test,evaluation.ceafe)
        print "CEAF: recall: %f precision: %f  f1: %f"%(cr,cp,cf)
        print 

        end_time = timeit.default_timer()
        print >> sys.stderr, "TEST Use %.3f seconds"%(end_time-start_time)
        sys.stdout.flush()

        '''
        save_f = file('./model/nets/network_model.%s.%d'%(args.language,echo), 'wb')
        cPickle.dump(network_model, save_f, protocol=cPickle.HIGHEST_PROTOCOL)
        save_f.close()
        '''

if __name__ == "__main__":
    main()
