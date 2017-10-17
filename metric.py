#coding=utf8
import os
import sys
import re
import argparse
import math
import timeit
import numpy
import random

from conf import *
import word2vec
import network
import generate_instance

import cPickle
sys.setrecursionlimit(1000000)

random.seed(args.random_seed)

def get_prf(anaphorics_result,predict_result):
    ## 如果 ZP 是负例 则没有anaphorics_result
    ## 如果 predict 出负例 则 predict_candi_sentence_index = -1
    should = 0
    right = 0
    predict_right = 0
    for i in range(len(predict_result)):
        (sentence_index,zp_index,predict_candi_sentence_index,predict_candi_begin,predict_candi_end) = predict_result[i]
        anaphoric = anaphorics_result[i] 
        if anaphoric:
            should += 1
            if (sentence_index,zp_index,predict_candi_sentence_index,predict_candi_begin,predict_candi_end) in anaphoric:
                right += 1
        if not (predict_candi_sentence_index == -1):
            predict_right += 1

    print "Should:",should,"Right:",right,"PredictRight:",predict_right
    if predict_right == 0:
        P = 0.0
    else:
        P = float(right)/float(predict_right)

    if should == 0:
        R = 0.0
    else:
        R = float(right)/float(should)

    if (R == 0.0) or (P == 0.0):
        F = 0.0
    else:
        F = 2.0/(1.0/P + 1.0/R)

    #print "P:",P
    #print "R:",R
    #print "F:",F

    return P,R,F

def get_sentence(zp_sentence_index,zp_index,nodes_info):
    #返回只包含zp_index位置的ZP的句子
    nl,wl = nodes_info[zp_sentence_index]
    return_words = []
    for i in range(len(wl)):
        this_word = wl[i].word
        if i == zp_index:
            return_words.append("**pro**")
        else:
            if not (this_word == "*pro*"): 
                return_words.append(this_word)
    return " ".join(return_words)

def get_candi_info(candi_sentence_index,nodes_info,candi_begin,candi_end,res_result):
    nl,wl = nodes_info[candi_sentence_index]
    candi_word = []
    for i in range(candi_begin,candi_end+1):
        candi_word.append(wl[i].word)
    candi_word = "_".join(candi_word)

    candi_info = [str(res_result),candi_word]
    return candi_info

def find_max(l):
    ### 找到list中最大的 返回index
    return_index = len(l)-1
    max_num = 0.0
    for i in range(len(l)):
        if l[i] >= max_num:
            max_num = l[i] 
            return_index = i
    return return_index

