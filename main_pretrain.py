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
#import policy_network
import policy_network_single as policy_network
import network
#import network_batch as network
import pretrain

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
    #net_dir = "./model/pretrain/network_model_pretrain.cn.19"
    #net_dir = "./model/pretrain_manu_dropout/network_model_pretrain.cn.10"
    if os.path.isfile("./model/network_model."+args.language):
        read_f = file('./model/network_model.'+args.language, 'rb')
        #read_f = file('./model/network_model_pretrain.'+args.language, 'rb')
        #read_f = file('./model/network_model_pretrain.cn.best', 'rb')
        #read_f = file(net_dir, 'rb')
        network_model = cPickle.load(read_f)
        print >> sys.stderr,"Read model from ./model/network_model."+args.language
    else:
        inpt_dimention = 1738
        single_dimention = 855
        if args.language == "en":
            inpt_dimention = 1374
            single_dimention = 673

        network_model = network.NetWork(inpt_dimention,single_dimention,1000)
        print >> sys.stderr,"save model ..."
        save_f = file('./model/network_model.'+args.language, 'wb')
        cPickle.dump(network_model, save_f, protocol=cPickle.HIGHEST_PROTOCOL)
        save_f.close()

    train_docs = DataGenerate.doc_data_generater("train")
    dev_docs = DataGenerate.doc_data_generater("dev")
    test_docs = DataGenerate.doc_data_generater("test")

    #pretrain
    l2_lambda = 0.0000001
    lr = 0.03
    ce_lambda = 0.0001
    dropout_rate = 0.2

    print "Weight Sum",network_model.get_weight_sum()

    times = 0
    #for echo in range(11,40):
    for echo in range(30):

        start_time = timeit.default_timer()
        print "Pretrain ECHO:",echo
        cost_this_turn = 0.0
        #print >> sys.stderr, network_model.get_weight_sum()
        done_num = 0
        pos_num = 0
        neg_num = 0
        for cases,gold_chain in DataGenerate.case_generater(train_docs,"train",w2v):
            if len(cases) >= 700:
                continue
            for single_mention_array,train_list,lable_list in pretrain.generate_pretrain_case(cases,gold_chain,network_model):
                cost_this_turn += network_model.pre_train_step(single_mention_array,train_list,lable_list,lr,l2_lambda,dropout_rate)[0]
                #cost_this_turn += network_model.pre_top_train_step(single_mention_array,train_list,lable_list,lr,l2_lambda)[0]

                if lable_list[0] == 1:
                    neg_num += 1
                else:
                    pos_num += 1

            done_num += 1
            #if done_num == 10:
            #    break
        lr = lr*0.99

        save_f = file('./model/pretrain_manu_new/network_model_pretrain.%s.%d'%(args.language,echo), 'wb')
        cPickle.dump(network_model, save_f, protocol=cPickle.HIGHEST_PROTOCOL)
        save_f.close()

        end_time = timeit.default_timer()
        print >> sys.stderr, "PreTrain",echo,"Total cost:",cost_this_turn
        print >> sys.stderr, "POS:NEG",pos_num,neg_num
        print >> sys.stderr, "lr",lr
        print >> sys.stderr, "PreTRAINING Use %.3f seconds"%(end_time-start_time)
        print "Weight Sum",network_model.get_weight_sum()

        ## test performance after pretraining
        dev_docs_for_test = []
        num = 0
        for cases,gold_chain in DataGenerate.case_generater(dev_docs,"dev",w2v):
            ev_doc = policy_network.generate_policy_test(cases,gold_chain,network_model)
            dev_docs_for_test.append(ev_doc)
            num += 1
            #if num == 10:
            #    break
        print "Performance on DEV after PreTRAINING"
        mp,mr,mf = evaluation.evaluate_documents(dev_docs_for_test,evaluation.muc)
        print "MUC: recall: %f precision: %f  f1: %f"%(mr,mp,mf)
        bp,br,bf = evaluation.evaluate_documents(dev_docs_for_test,evaluation.b_cubed)
        print "BCUBED: recall: %f precision: %f  f1: %f"%(br,bp,bf)
        cp,cr,cf = evaluation.evaluate_documents(dev_docs_for_test,evaluation.ceafe)
        print "CEAF: recall: %f precision: %f  f1: %f"%(cr,cp,cf)
        print "#################################################" 
        sys.stdout.flush()
    return

    for echo in range(30,50):
        start_time = timeit.default_timer()
        cost_this_turn = 0.0
        for cases,gold_chain in DataGenerate.case_generater(train_docs,"train",w2v):
            if len(cases) >= 700:
                continue
            for single_mention_array,train_list,lable_list in pretrain.generate_pretrain_case(cases,gold_chain,network_model):
                cost_this_turn += network_model.pre_ce_train_step(single_mention_array,train_list,lable_list,lr,l2_lambda,ce_lambda)[0]

        end_time = timeit.default_timer()
        print >> sys.stderr, "PreTrain",echo,"Total cost:",cost_this_turn
        print >> sys.stderr, "PreTRAINING Use %.3f seconds"%(end_time-start_time)
        print "Weight Sum",network_model.get_weight_sum()

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
        print "#################################################" 
        sys.stdout.flush()

        save_f = file('./model/pretrain_manu_new/network_model_pretrain.%s.%d'%(args.language,echo), 'wb')
        cPickle.dump(network_model, save_f, protocol=cPickle.HIGHEST_PROTOCOL)
        save_f.close()


    ## test performance after pretraining
    print >> sys.stderr,"Begin test on DEV after pertraining"
    dev_docs_for_test = []
    num = 0
    #for dev_doc_mention_array,dev_doc_pair_array,dev_doc_gold_chain in DataGenerate.array_generater(dev_docs,"dev",w2v):
        #ev_doc = policy_network.generate_policy_test(dev_doc_mention_array,dev_doc_pair_array,dev_doc_gold_chain,network_model)
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
    print "#################################################" 
    sys.stdout.flush()
    print >> sys.stderr,"Pre Train done"
if __name__ == "__main__":
    main()
