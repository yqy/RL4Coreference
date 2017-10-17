#coding=utf8
import os
import sys
import re
import argparse
import math
import timeit
import numpy
import random
from subprocess import *

from conf import *
from buildTree import get_info_from_file
from buildTree import get_info_from_file_system
import get_dir
import get_feature
import word2vec
import network
import generate_instance
from metric import *


import cPickle
sys.setrecursionlimit(1000000)

if(len(sys.argv) <= 1): 
    sys.stderr.write("Not specify options, type '-h' for help\n")
    exit()

print >> sys.stderr, os.getpid()

random.seed(args.random_seed)

if args.type == "nn_train":

    if os.path.isfile("./model/save_data"):
        print >> sys.stderr,"Read from file ./model/save_data"
        read_f = file('./model/save_data', 'rb')        
        training_instances = cPickle.load(read_f)
        anaphorics_result = cPickle.load(read_f)
        test_instances = cPickle.load(read_f)
        read_f.close()
    else:
        print >> sys.stderr, "Read W2V"
        w2v = word2vec.Word2Vec(args.embedding)
        
        ### Training ####    
        path = args.data
        #training_instances = generate_instance.generate_training_instances_feature(path,w2v)
        training_instances = generate_instance.generate_training_instances(path,w2v)
    
        ####  Test process  ####
    
        path = args.test_data
        #test_instances,anaphorics_result = generate_instance.generate_test_instances_feature(path,w2v)
        test_instances,anaphorics_result = generate_instance.generate_test_instances(path,w2v)

        w2v = None # 释放空间
        print >> sys.stderr,"Save file ./model/save_data"

        save_f = file('./model/save_data', 'wb')
        cPickle.dump(training_instances, save_f, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(anaphorics_result, save_f, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(test_instances, save_f, protocol=cPickle.HIGHEST_PROTOCOL)
        save_f.close()

    
    ## Build Neural Network Model ## 
    print >> sys.stderr,"Building Model ..."
    
    if os.path.isfile("./model/lstm_init_model"):
        read_f = file('./model/lstm_init_model', 'rb')
        LSTM = cPickle.load(read_f)
        print >> sys.stderr,"Read model from ./model/lstm_init_model"
    else: 
        LSTM = network.NetWork(128,args.embedding_dimention,61)
        #LSTM = network.NetWork(128,args.embedding_dimention,84)
        print >> sys.stderr,"save model ..."
        save_f = file('./model/lstm_init_model', 'wb') 
        cPickle.dump(LSTM, save_f, protocol=cPickle.HIGHEST_PROTOCOL)
        save_f.close()

    ##### begin train and test #####

    DEV_PROB = args.dev_prob
    random.shuffle(training_instances)

    dev_instances = training_instances[:int(DEV_PROB*len(training_instances))]
    train_instances = training_instances[int(DEV_PROB*len(training_instances))+1:]

    train_test_instances = train_instances[:int(DEV_PROB*len(training_instances))] # 提取出一部分train data 看在train的效果

    dev_hits = 0
    dev_iteration = 0
    test_hits_in_dev = 0 ## 当dev hits 最多的时候 test hits 数目

    test_hits = 0
    test_iteration = 0 
    
    for echo in range(args.echos): 
        print >> sys.stderr, "Echo for time",echo

        start_time = timeit.default_timer()
        cost = 0.0

        random.shuffle(train_instances)

        #for zp_x_pre,zp_x_post,np_x_pre_list,np_x_prec_list,np_x_post_list,np_x_postc_list,mask_pre,mask_prec,mask_post,mask_postc,feature_list,res_list in train_instances:  # with feature
        for zp_x_pre,zp_x_post,np_x_pre_list,np_x_prec_list,np_x_post_list,np_x_postc_list,mask_pre,mask_prec,mask_post,mask_postc,res_list in train_instances:
            #cost += LSTM.train_step(zp_x_pre,zp_x_post,np_x_pre_list,np_x_prec_list,np_x_post_list,np_x_postc_list,mask_pre,mask_prec,mask_post,mask_postc,feature_list,res_list,args.lr,args.dropout_prob)[0]
            cost += LSTM.train_step(zp_x_pre,zp_x_post,np_x_pre_list,np_x_prec_list,np_x_post_list,np_x_postc_list,mask_pre,mask_prec,mask_post,mask_postc,res_list,args.lr,args.dropout_prob)[0]

        end_time = timeit.default_timer()
        print >> sys.stderr,"Cost",cost
        print >> sys.stderr,"Parameters"
        LSTM.show_para()
        print >> sys.stderr, end_time - start_time, "seconds!"

        #### Test for each echo ####
        print >> sys.stderr, "Begin test" 
        predict_result = []
        numOfZP = 0
        test_hits_this = 0
        #for (zp_x_pre,zp_x_post,np_x_pre_list,np_x_prec_list,np_x_post_list,np_x_postc_list,mask_pre,mask_prec,mask_post,mask_postc,feature_list,res_list,zp_candi_list,nodes_info) in test_instances:
        for (zp_x_pre,zp_x_post,np_x_pre_list,np_x_prec_list,np_x_post_list,np_x_postc_list,mask_pre,mask_prec,mask_post,mask_postc,res_list,zp_candi_list,nodes_info) in test_instances:

            numOfZP += 1
            if len(np_x_pre_list) == 0: ## no suitable candidates
                predict_result.append((-1,-1,-1,-1,-1))
            else:
                zp,candidate = zp_candi_list[-1]
                sentence_index,zp_index = zp
                print >> sys.stderr,"------" 
                this_sentence = get_sentence(sentence_index,zp_index,nodes_info)
                print >> sys.stderr, "Sentence:",this_sentence

                print >> sys.stderr, "Candidates:"

                #outputs = list(LSTM.get_out(zp_x_pre,zp_x_post,np_x_pre_list,np_x_prec_list,np_x_post_list,np_x_postc_list,mask_pre,mask_prec,mask_post,mask_postc,feature_list,0.0)[0])
                outputs = list(LSTM.get_out(zp_x_pre,zp_x_post,np_x_pre_list,np_x_prec_list,np_x_post_list,np_x_postc_list,mask_pre,mask_prec,mask_post,mask_postc,0.0)[0])
                max_index = find_max(outputs)
                if res_list[max_index] == 1:
                    test_hits_this += 1

                st_score = 0.0
                predict_items = None
                numOfCandi = 0
                predict_str_log = None
                for i in range(len(zp_candi_list)): 
                    zp,candidate = zp_candi_list[i]
                    nn_predict = outputs[i]
                    res_result = res_list[i]
                
                    candi_sentence_index,candi_begin,candi_end = candidate
                    candi_str = "\t".join(get_candi_info(candi_sentence_index,nodes_info,candi_begin,candi_end,res_result))
                    if nn_predict >= st_score: 
                        predict_items = (zp,candidate)
                        st_score = nn_predict
                        predict_str_log = "%d\t%s\tPredict:%f"%(numOfCandi,candi_str,nn_predict)
                    print >> sys.stderr,"%d\t%s\tPredict:%f"%(numOfCandi,candi_str,nn_predict)
                    numOfCandi += 1

                predict_zp,predict_candidate = predict_items
                sentence_index,zp_index = predict_zp 
                predict_candi_sentence_index,predict_candi_begin,predict_candi_end = predict_candidate

                predict_result.append((sentence_index,zp_index,predict_candi_sentence_index,predict_candi_begin,predict_candi_end))
                print >> sys.stderr, "Results:"
                print >> sys.stderr, "Predict -- %s"%predict_str_log
                print >> sys.stderr, "Done ZP #%d/%d"%(numOfZP,len(test_instances))

        print >> sys.stderr, "Test Hits:",test_hits_this,"/",len(test_instances)," in iteration", echo

        if test_hits_this >= test_hits:
            test_hits = test_hits_this
            test_iteration = echo

        ### see how many hits in DEV ###
        dev_hits_this = 0
        #for zp_x_pre,zp_x_post,np_x_pre_list,np_x_prec_list,np_x_post_list,np_x_postc_list,mask_pre,mask_prec,mask_post,mask_postc,feature_list,res_list in dev_instances:
        for zp_x_pre,zp_x_post,np_x_pre_list,np_x_prec_list,np_x_post_list,np_x_postc_list,mask_pre,mask_prec,mask_post,mask_postc,res_list in dev_instances:
            #outputs = list(LSTM.get_out(zp_x_pre,zp_x_post,np_x_pre_list,np_x_prec_list,np_x_post_list,np_x_postc_list,mask_pre,mask_prec,mask_post,mask_postc,feature_list,0.0)[0])
            outputs = list(LSTM.get_out(zp_x_pre,zp_x_post,np_x_pre_list,np_x_prec_list,np_x_post_list,np_x_postc_list,mask_pre,mask_prec,mask_post,mask_postc,0.0)[0])
            max_index = find_max(outputs)
            if res_list[max_index] == 1:
                dev_hits_this += 1 

        print >> sys.stderr, "DEV Hits:",dev_hits_this,"/",len(dev_instances)," in iteration", echo
        if dev_hits_this >= dev_hits:
            dev_hits = dev_hits_this
            dev_iteration = echo
            test_hits_in_dev = test_hits_this

        ### see how many hits in TRAIN ###
        train_hits_this = 0
        for zp_x_pre,zp_x_post,np_x_pre_list,np_x_prec_list,np_x_post_list,np_x_postc_list,mask_pre,mask_prec,mask_post,mask_postc,res_list in train_test_instances:
            outputs = list(LSTM.get_out(zp_x_pre,zp_x_post,np_x_pre_list,np_x_prec_list,np_x_post_list,np_x_postc_list,mask_pre,mask_prec,mask_post,mask_postc,0.0)[0])
            max_index = find_max(outputs)
            if res_list[max_index] == 1:
                train_hits_this += 1 

        print >> sys.stderr, "TRAIN Hits:",train_hits_this,"/",len(train_test_instances)," in iteration", echo

        print >> sys.stderr,"Summary: best hits in dev is %d in iteration %d (test hits %d) --- best hits in test is %d in iteration %d"%(dev_hits,dev_iteration,test_hits_in_dev,test_hits,test_iteration)

        print "Echo",echo 
        print "Test Hits:",test_hits_this,"/",len(test_instances)
        P,R,F = get_prf(anaphorics_result,predict_result)
        print "P:",P
        print "R:",R
        print "F:",F

        sys.stdout.flush()

    print >> sys.stderr,"Over for all"
