#coding=utf8
import sys
from numpy import linalg
from numpy import array
from numpy import inner

import pickle

import cPickle
sys.setrecursionlimit(1000000)

def cosin(Al,Bl):
    #Al,Bl 向量的list
    A = array(Al)
    B = array(Bl)
    nA = linalg.norm(A)
    nB = linalg.norm(B)
    if nA == 0 or nB == 0:
        return 0
    #num = float(A * B.T) #行向量
    num = inner(A,B)
    denom = nA * nB
    cos = num / denom #余弦值
    #sim = 0.5 + 0.5 * cos #归一化
    sim = cos
    return sim

class Word2Vec:
    word_dict = {}
    index_dict = {}
    def __init__(self,w2v_dir="",dimention=50):
        if w2v_dir == "":
            print >> sys.stderr, "Please give the embedding file"
            return
        self.dimention = dimention
        f = file(w2v_dir, 'rb')
        embedding_list = cPickle.load(f)
        num = 1
        for word,em in embedding_list:
            self.word_dict[word] = em
            self.index_dict[word] = num
            num += 1
        print >> sys.stderr,"Total %d word embedding!"%len(embedding_list)
    def get_vector_by_word(self,word):
        if word in self.word_dict:
            return self.word_dict[word]
        else:
            return array([0.0]*self.dimention)

    def get_index_by_word(self,word):
        if word in self.index_dict:
            return self.index_dict[word]
        else:
            return 0

    def get_vector_by_list(self,wl):
        result = array([0.0]*self.dimention)
        for word in wl:
            result += self.get_vector_by_word(word)
        if len(wl) == 0:
            return array([0.0]*self.dimention)
        return result/len(wl)

def main():
    #w2v = Word2Vec("/Users/yqy/data/coreference/embedding/embedding.en.filtered",50)
    w2v = Word2Vec("/Users/yqy/data/coreference/embedding/embedding.cn.filtered",64)
    print "go"
    while True:
        line = sys.stdin.readline()
        if not line:break
        line = line.strip().split(" ")
        l1 = w2v.get_vector_by_word(line[0].strip())
        l2 = w2v.get_vector_by_word(line[1].strip())
        print l1
        print l2
        print cosin(l1,l2)

def generate_embedding_file_from_Kevin():
    # generate the word embedding file from Kevin's github file
    # https://github.com/clarkkev/deep-coref
    # "We use 50 dimensional word2vec embeddings for English (link) and 64 dimenensional polyglot embeddings for Chinese (link) in our paper."
    # for english embedding file

    f = open("/Users/yqy/Downloads/w2v_50d.txt")
    embedding_list = []
    while True:
        line = f.readline()
        if not line:break
        line = line.strip().split("\t")
        word = line[0]
        embedding = array([ float(x) for x in line[1].split(" ")])
        embedding_list.append((word,embedding))
    save_f = file('./embedding.en', 'wb')
    cPickle.dump(embedding_list, save_f, protocol=cPickle.HIGHEST_PROTOCOL)

    # for chinese embedding file
    words, embeddings = pickle.load(open('/Users/yqy/Downloads/polyglot-zh.pkl', 'rb'))
    embedding_list = []
    for i in range(len(words)):
        word = words[i].encode("utf8")
        embedding = embeddings[i]
        embedding_list.append((word,embedding))
    save_f = file('./embedding.cn', 'wb')
    cPickle.dump(embedding_list, save_f, protocol=cPickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
