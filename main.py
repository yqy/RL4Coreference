#coding=utf8

import sys
import os
import json
import random
import numpy

from conf import *

import Mention
import Reader
import word2vec

import cPickle
sys.setrecursionlimit(1000000)

print >> sys.stderr, os.getpid()

random.seed(args.random_seed)

def get_doc_data():

    print >> sys.stderr,"Read and Generate TRAINING data ..."
    if os.path.isfile("./model/train_docs."+args.language):
        print >> sys.stderr,"Read train data from ./model/train_docs."+args.language
        read_f = file('./model/train_docs.'+args.language, 'rb')
        train_docs = cPickle.load(read_f)
    else:
        train_docs = Reader.read_from_file(args.train_data,args.train_gold,args.language)
        print >> sys.stderr,"save model ..."
        save_f = file('./model/train_docs.'+args.language, 'wb')
        cPickle.dump(train_docs, save_f, protocol=cPickle.HIGHEST_PROTOCOL)
        save_f.close()

    print >> sys.stderr,"Read and Generate DEV data ..."
    if os.path.isfile("./model/dev_docs."+args.language):
        print >> sys.stderr,"Read dev data from ./model/dev_docs."+args.language
        read_f = file('./model/dev_docs.'+args.language, 'rb')
        dev_docs = cPickle.load(read_f)
    else:
        dev_docs = Reader.read_from_file(args.dev_data,args.dev_gold,args.language)
        print >> sys.stderr,"save model ..."
        save_f = file('./model/dev_docs.'+args.language, 'wb')
        cPickle.dump(dev_docs, save_f, protocol=cPickle.HIGHEST_PROTOCOL)
        save_f.close()

    print >> sys.stderr,"Read and Generate TEST data ..."
    if os.path.isfile("./model/test_docs."+args.language):
        print >> sys.stderr,"Read test data from ./model/test_docs."+args.language
        read_f = file('./model/test_docs.'+args.language, 'rb')
        test_docs = cPickle.load(read_f)
    else:
        test_docs = Reader.read_from_file(args.test_data,args.test_gold,args.language)
        print >> sys.stderr,"save model ..."
        save_f = file('./model/test_docs.'+args.language, 'wb')
        cPickle.dump(test_docs, save_f, protocol=cPickle.HIGHEST_PROTOCOL)
        save_f.close()
    
    return train_docs,dev_docs,test_docs

def get_binary_by_num(length):
    # return binary list that convernt length into 0,1,0,0,0 vectors
    # rules: [0,1,2,3,4,5-7,8-15,16-31,32-63,64+] 
    # total 10 items
    rl = [0]*10
    if length < 5:
        rl[length] = 1
    elif length <= 7:
        rl[5] = 1
    elif length <= 15:
        rl[6] = 1
    elif length <= 31:
        rl[7] = 1
    elif length <= 63:
        rl[8] = 1
    else:
        rl[9] = 1
    return rl

def get_embedding(mention,w2v,doc):

    embedding_array = numpy.array([])

    first_word_embedding = w2v.get_vector_by_word(mention.words[0])
    embedding_array = numpy.append(embedding_array,first_word_embedding)
    last_word_embedding = w2v.get_vector_by_word(mention.words[-1])
    embedding_array = numpy.append(embedding_array,last_word_embedding)
    head_word_embedding = w2v.get_vector_by_word(mention.sentence[mention.head_index])
    embedding_array = numpy.append(embedding_array,head_word_embedding)
    depend_word_embedding = w2v.get_vector_by_word(mention.dep_parent)
    embedding_array = numpy.append(embedding_array,depend_word_embedding)

    precedding_1_word_embedding = w2v.get_vector_by_word((mention.sentence[mention.start_index-1] if (mention.start_index-1 >= 0) else "shenmeyemeiyou"))
    embedding_array = numpy.append(embedding_array,precedding_1_word_embedding)
    precedding_2_word_embedding = w2v.get_vector_by_word((mention.sentence[mention.start_index-2] if (mention.start_index-2 >= 0) else "shenmeyemeiyou"))
    embedding_array = numpy.append(embedding_array,precedding_2_word_embedding)

    following_1_word_embedding = w2v.get_vector_by_word((mention.sentence[mention.end_index] if (mention.end_index < len(mention.sentence)) else "shenmeyemeiyou" ))
    embedding_array = numpy.append(embedding_array,following_1_word_embedding)
    following_2_word_embedding = w2v.get_vector_by_word((mention.sentence[mention.end_index+1] if (mention.end_index+1 < len(mention.sentence)) else "shenmeyemeiyou" ))
    embedding_array = numpy.append(embedding_array,following_2_word_embedding)

    ave_words_embedding = w2v.get_vector_by_list(mention.words)
    embedding_array = numpy.append(embedding_array,ave_words_embedding)

    ave_sentence_embedding = w2v.get_vector_by_list(mention.sentence)
    embedding_array = numpy.append(embedding_array,ave_sentence_embedding)

    ave_five_following_embedding = w2v.get_vector_by_list(mention.sentence[mention.end_index:mention.end_index+5])
    embedding_array = numpy.append(embedding_array,ave_five_following_embedding)

    ave_five_precedding_embedding = w2v.get_vector_by_list(mention.sentence[max(0,mention.start_index-5):mention.start_index])
    embedding_array = numpy.append(embedding_array,ave_five_precedding_embedding)

    ave_document_embedding = w2v.get_vector_by_list(doc.document_words)
    embedding_array = numpy.append(embedding_array,ave_document_embedding)

    feature_array = []
    mention_type = [0]*4
    if mention.mention_type == "NOMINAL":
        mention_type[0] = 1
    elif mention.mention_type == "PRONOMINAL":
        mention_type[1] = 1
    elif mention.mention_type == "PROPER":
        mention_type[2] = 1
    else:
        mention_type[3] = 1 
    feature_array += mention_type

    mention_position = float(mention.mention_num)/float(len(doc.mentions))
    feature_array += [mention_position]

    mention_contain_in_other = [0]
    if mention.contained_in_other_mention == 1:
        mention_contain_in_other = [1]
    feature_array += mention_contain_in_other

    mention_length = len(mention.words)
    feature_array += get_binary_by_num(mention_length)

    doc_type = [0]*6
    doc_dict = {"bn":0,"nw":1,"bc":2,"tc":3,"wb":4,"mz":5}
    doc_type[doc_dict[doc.doc_source]]=1
    feature_array += doc_type

    feature_array = numpy.array(feature_array) # dimention = 22

    return embedding_array,feature_array 

def get_pair_embedding(i,j,doc):
    feature_array = []

    mention_i = doc.mentions[i]
    mention_j = doc.mentions[j]
    pair_feature = list(doc.pair_feature[(i,j)])
    feature_array += pair_feature

    words_i = " ".join(mention_i.words)
    words_j = " ".join(mention_j.words)
    overlap = [0]*2
    if words_i in words_j:
        overlap[0] = 1
    if words_j in words_i:
        overlap[1] = 1
    feature_array += overlap

    # distance feature
    sentence_distance = max(mention_i.sent_num,mention_j.sent_num) - min(mention_i.sent_num,mention_j.sent_num)
    feature_array += get_binary_by_num(sentence_distance)

    mention_distance = max(mention_i.mention_num,mention_j.mention_num) - min(mention_i.mention_num,mention_j.mention_num)
    feature_array += get_binary_by_num(sentence_distance)

    feature_array = numpy.array(feature_array) # dimention=28
    return feature_array

def main():

    embedding_dir = args.embedding+args.language
    print >> sys.stderr,"Read Embedding from %s ..."%embedding_dir
    embedding_dimention = 50
    if args.language == "cn":
        embedding_dimention = 64
    w2v = word2vec.Word2Vec(embedding_dir,embedding_dimention)

    train_docs,dev_docs,test_docs = get_doc_data()

    ## generate training inputs
    for doc in train_docs:
        mention_arrays = []
        pair_arrays = {}

        start_time = timeit.default_timer()
        
        ## feature and embedding for each Mention
        for mention in doc.mentions:
            this_mention_embedding,this_mention_feature = get_embedding(mention,w2v,doc)
            mention_arrays.append((this_mention_embedding,this_mention_feature))

        ## features for each pair
        for i in range(len(doc.mentions)):
            for j in range(i+1,len(doc.mentions)):
                pair_feature = get_pair_embedding(i,j,doc)
                pair_arrays[(i,j)] = pair_feature

        print >> sys.stderr,"SV mention_array_data to ./model/mention_array_train."+args.language
        save_f = file('./model/mention_array_train.'+args.language, 'wb')
        cPickle.dump((mention_arrays,pair_arrays), save_f, protocol=cPickle.HIGHEST_PROTOCOL)
        save_f.close()
        end_time = timeit.default_timer()
        print >> sys.stderr, "Use %.3f seconds"%(end_time-start_time)

    ## generate dev inputs
    for doc in dev_docs:
        mention_arrays = []
        pair_arrays = {}

        start_time = timeit.default_timer()
        
        ## feature and embedding for each Mention
        for mention in doc.mentions:
            this_mention_embedding,this_mention_feature = get_embedding(mention,w2v,doc)
            mention_arrays.append((this_mention_embedding,this_mention_feature))

        ## features for each pair
        for i in range(len(doc.mentions)):
            for j in range(i+1,len(doc.mentions)):
                pair_feature = get_pair_embedding(i,j,doc)
                pair_arrays[(i,j)] = pair_feature

        print >> sys.stderr,"SV mention_array_data to ./model/mention_array_dev."+args.language
        save_f = file('./model/mention_array_dev.'+args.language, 'wb')
        cPickle.dump((mention_arrays,pair_arrays), save_f, protocol=cPickle.HIGHEST_PROTOCOL)
        save_f.close()
        end_time = timeit.default_timer()
        print >> sys.stderr, "Use %.3f seconds"%(end_time-start_time)


    ## generate test inputs
    for doc in test_docs:
        mention_arrays = []
        pair_arrays = {}

        start_time = timeit.default_timer()
        
        ## feature and embedding for each Mention
        for mention in doc.mentions:
            this_mention_embedding,this_mention_feature = get_embedding(mention,w2v,doc)
            mention_arrays.append((this_mention_embedding,this_mention_feature))

        ## features for each pair
        for i in range(len(doc.mentions)):
            for j in range(i+1,len(doc.mentions)):
                pair_feature = get_pair_embedding(i,j,doc)
                pair_arrays[(i,j)] = pair_feature

        print >> sys.stderr,"SV mention_array_data to ./model/mention_array_test."+args.language
        save_f = file('./model/mention_array_test.'+args.language, 'wb')
        cPickle.dump((mention_arrays,pair_arrays), save_f, protocol=cPickle.HIGHEST_PROTOCOL)
        save_f.close()
        end_time = timeit.default_timer()
        print >> sys.stderr, "Use %.3f seconds"%(end_time-start_time)


        
if __name__ == "__main__":
    main()
