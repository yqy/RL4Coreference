#coding=utf8

import sys
import os
import json
import random
import numpy
import timeit

from conf import *

import Mention
#import Reader
import word2vec

import cPickle
sys.setrecursionlimit(1000000)

ENDTAG="EOF!!!EOF"

def read_from_file(fil,goldf,language):
    line_num = len(open(goldf).readlines())
    print >> sys.stderr, "Read from %s and %s documents"%(fil,goldf)

    f = open(fil)
    gold_file = open(goldf)

    total_start_time = timeit.default_timer()
    i = 0
    while True:
        line = f.readline()
        if not line:break
        i += 1

        #if i >= 5:
        #    break 

        start_time = timeit.default_timer()
        line = line.strip()
        s = json.loads(line)
        line = gold_file.readline()
        js = json.loads(line)
        gold_chain = js[js.keys()[0]]

        document = Mention.Document(s,gold_chain,language)
        yield document

        end_time = timeit.default_timer()
        print >> sys.stderr, "Done %d/%d document - %.3f seconds"%(i,line_num,end_time-start_time)

    total_end_time = timeit.default_timer()
    print >> sys.stderr, "Total use %.3f seconds"%(total_end_time-total_start_time)

def doc_data_generater(doc_type):

    if doc_type == "train":
        print >> sys.stderr,"Read and Generate TRAINING data ..."
        if os.path.isfile("./model/train_docs."+args.language):
            print >> sys.stderr,"Read train data from ./model/train_docs."+args.language
            read_f = file('./model/train_docs.'+args.language, 'rb')
            while True:
                train_doc = cPickle.load(read_f)
                if train_doc == ENDTAG:
                    break
                yield train_doc
        else:
            train_docs = read_from_file(args.train_data,args.train_gold,args.language)
            print >> sys.stderr,"save model ..."
            save_f = file('./model/train_docs.'+args.language, 'wb')
            for train_doc in train_docs:
                cPickle.dump(train_doc, save_f, protocol=cPickle.HIGHEST_PROTOCOL)
                yield train_doc
            cPickle.dump(ENDTAG, save_f, protocol=cPickle.HIGHEST_PROTOCOL)
            save_f.close()

    elif doc_type == "dev":
        print >> sys.stderr,"Read and Generate DEV data ..."
        if os.path.isfile("./model/dev_docs."+args.language):
            print >> sys.stderr,"Read dev data from ./model/dev_docs."+args.language
            read_f = file('./model/dev_docs.'+args.language, 'rb')
            while True:
                dev_doc = cPickle.load(read_f)
                if dev_doc == ENDTAG:
                    break
                yield dev_doc
        else:
            dev_docs = read_from_file(args.dev_data,args.dev_gold,args.language)
            print >> sys.stderr,"save model ..."
            save_f = file('./model/dev_docs.'+args.language, 'wb')
            for dev_doc in dev_docs: 
                cPickle.dump(dev_doc, save_f, protocol=cPickle.HIGHEST_PROTOCOL)
                yield dev_doc
            cPickle.dump(ENDTAG, save_f, protocol=cPickle.HIGHEST_PROTOCOL)
            save_f.close()
    elif doc_type == "test":
        print >> sys.stderr,"Read and Generate TEST data ..."
        if os.path.isfile("./model/test_docs."+args.language):
            print >> sys.stderr,"Read test data from ./model/test_docs."+args.language
            read_f = file('./model/test_docs.'+args.language, 'rb')
            while True:
                test_doc = cPickle.load(read_f)
                if test_doc == ENDTAG:
                    break
                yield test_doc
        else:
            test_docs = read_from_file(args.test_data,args.test_gold,args.language)
            print >> sys.stderr,"save model ..."
            save_f = file('./model/test_docs.'+args.language, 'wb')
            for test_doc in test_docs:
                cPickle.dump(test_doc, save_f, protocol=cPickle.HIGHEST_PROTOCOL)
                yield test_doc
            cPickle.dump(ENDTAG, save_f, protocol=cPickle.HIGHEST_PROTOCOL)
            save_f.close()

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

    doc_type = [0]*7
    doc_dict = {"bn":0,"nw":1,"bc":2,"tc":3,"wb":4,"mz":5,"pt":6}
    doc_type[doc_dict[doc.doc_source]]=1
    feature_array += doc_type

    feature_array = numpy.array(feature_array) # dimention = 23

    return numpy.append(embedding_array,feature_array)

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

    feature_array = numpy.array(feature_array,dtype=numpy.float32) # dimention=28
    return feature_array

def generate_input_case(doc_mention_arrays,doc_pair_arrays):

    train_case = []

    mentions_num = len(doc_mention_arrays)

    for i in range(mentions_num):
        mention_array = doc_mention_arrays[i]
        this_train_case = []

        for j in range(0,i):
            mention_in_cluster_array = doc_mention_arrays[j]
            #pair_features = doc_pair_arrays[(2*mentions_num-j-1)*j/2 + i-j -1]  #等差数列算出
            pair_features = doc_pair_arrays[(j,i)]
            this_input = numpy.append(mention_array,mention_in_cluster_array)
            this_input = numpy.append(this_input,pair_features)
            this_train_case.append(this_input)

        this_train_case = numpy.array(this_train_case,dtype = numpy.float32)

        mention_array_float = numpy.array([mention_array],dtype = numpy.float32)

        train_case.append((mention_array_float,this_train_case))

    return train_case

def case_generater(docs,typ,w2v):
    for mention_arrays, pair_arrays, gold_chain in array_generater(docs,typ,w2v):
        if typ == "train":
            if len(mention_arrays) >= 700:
                continue
        train_case = generate_input_case(mention_arrays,pair_arrays)
        yield train_case,gold_chain


def case_generater_save(docs,typ,w2v):

    if os.path.isfile("./model/case_input_%s."%typ+args.language):
        print >> sys.stderr,"Read data from ./model/case_input_%s."%typ+args.language
        read_f = file("./model/case_input_%s."%typ+args.language, 'rb')
        doc_num = 1
        while True:
            start_time = timeit.default_timer()
            train_case,gold_chain = cPickle.load(read_f)
            #doc_mention_array,doc_pair_array,doc_gold_chain = cPickle.load(read_f)
            if train_case == ENDTAG:
                break
            yield train_case,gold_chain
            end_time = timeit.default_timer()
            print >> sys.stderr, "Use %.3f seconds for doc %d with %d mentions"%(end_time-start_time,doc_num,len(train_case))
            doc_num += 1
    else:
        print >> sys.stderr,"Generate %s cases inputs for %s"%(typ,args.language)

        start_time = timeit.default_timer()

        print >> sys.stderr,"SV case input to ./model/case_input_%s."%typ+args.language
        save_f = file('./model/case_input_%s.'%typ+args.language, 'wb')
    
        for doc in docs:
            mention_arrays = []
            #pair_arrays = []
            pair_arrays = {}

            ## feature and embedding for each Mention
            for mention in doc.mentions:
                this_mention_embedding = get_embedding(mention,w2v,doc)
                mention_arrays.append(this_mention_embedding)
    
            ## features for each pair
            mentions_nums = len(doc.mentions)
            for i in range(len(doc.mentions)):
                for j in range(i+1,len(doc.mentions)):
                    pair_feature = get_pair_embedding(i,j,doc)
                    #pair_arrays.append(pair_feature) # i,j : pair_arrays[i*mentions_nums+j] = pair_feature
                    pair_arrays[(i,j)] = pair_feature

            train_case = generate_input_case(numpy.array(mention_arrays,dtype = numpy.float32),pair_arrays)

            #cPickle.dump((mention_arrays,pair_arrays,doc.gold_chain), save_f, protocol=cPickle.HIGHEST_PROTOCOL)
            cPickle.dump((train_case,doc.gold_chain), save_f, protocol=cPickle.HIGHEST_PROTOCOL)

            yield train_case,doc.gold_chain
   
        cPickle.dump((ENDTAG,None), save_f, protocol=cPickle.HIGHEST_PROTOCOL)
        save_f.close()


def array_generater(docs,typ,w2v):

    if os.path.isfile("./model/mention_array_%s."%typ+args.language):
        print >> sys.stderr,"Read data from ./model/mention_array_%s."%typ+args.language
        read_f = file("./model/mention_array_%s."%typ+args.language, 'rb')
        doc_num = 1
        while True:
            start_time = timeit.default_timer()
            doc_mention_array,doc_pair_array,doc_gold_chain = cPickle.load(read_f)
            if doc_mention_array == ENDTAG:
                break
            yield doc_mention_array,doc_pair_array,doc_gold_chain
            end_time = timeit.default_timer()
            print >> sys.stderr, "Use %.3f seconds for doc %d with %d mentions"%(end_time-start_time,doc_num,len(doc_mention_array))
            doc_num += 1
    else:
        print >> sys.stderr,"Generate %s arrays for %s"%(typ,args.language)

        start_time = timeit.default_timer()

        print >> sys.stderr,"SV mention_array_data to ./model/mention_array_%s."%typ+args.language
        save_f = file('./model/mention_array_%s.'%typ+args.language, 'wb')
    
        for doc in docs:
            mention_arrays = []
            #pair_arrays = []
            pair_arrays = {}

    
            ## feature and embedding for each Mention
            for mention in doc.mentions:
                this_mention_embedding = get_embedding(mention,w2v,doc)
                mention_arrays.append(this_mention_embedding)
    
            ## features for each pair
            mentions_nums = len(doc.mentions)
            for i in range(len(doc.mentions)):
                for j in range(i+1,len(doc.mentions)):
                    pair_feature = get_pair_embedding(i,j,doc)
                    #pair_arrays.append(pair_feature) # i,j : pair_arrays[i*mentions_nums+j] = pair_feature
                    pair_arrays[(i,j)] = pair_feature

            cPickle.dump((mention_arrays,pair_arrays,doc.gold_chain), save_f, protocol=cPickle.HIGHEST_PROTOCOL)

            #yield numpy.array(mention_arrays),numpy.array(pair_arrays),doc.gold_chain
            yield numpy.array(mention_arrays),pair_arrays,doc.gold_chain
   
        cPickle.dump((ENDTAG,None,None), save_f, protocol=cPickle.HIGHEST_PROTOCOL)
        save_f.close()


def main():

    embedding_dir = args.embedding+args.language
    print >> sys.stderr,"Read Embedding from %s ..."%embedding_dir
    embedding_dimention = 50
    if args.language == "cn":
        embedding_dimention = 64
    w2v = word2vec.Word2Vec(embedding_dir,embedding_dimention)

    #train_docs,dev_docs,test_docs = get_doc_data()
    train_docs = doc_data_generater("train")
    dev_docs = doc_data_generater("dev")
    test_docs = doc_data_generater("test")

    train_doc_mention_arrays,train_doc_pair_arrays,train_doc_gold_chains = array_generater(train_docs,"train",w2v)
    test_doc_mention_arrays,test_doc_pair_arrays,test_doc_gold_chains = array_generater(test_docs,"test",w2v)
    dev_doc_mention_arrays,dev_doc_pair_arrays,dev_doc_gold_chains = array_generater(dev_docs,"dev",w2v)

if __name__ == "__main__":
    main()
