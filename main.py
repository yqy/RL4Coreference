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
    test_doc_mention_arrays,test_doc_pair_arrays = DataGenerate.get_arrays(test_docs,"test",w2v)
    dev_doc_mention_arrays,dev_doc_pair_arrays = DataGenerate.get_arrays(dev_docs,"dev",w2v)

if __name__ == "__main__":
    main()
