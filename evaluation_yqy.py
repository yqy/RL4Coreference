import sys
import numpy as np
from collections import Counter
from sklearn.utils.linear_assignment_ import linear_assignment


def f1(p_num, p_den, r_num, r_den, beta=1):
    p = 0 if p_den == 0 else p_num / float(p_den)
    r = 0 if r_den == 0 else r_num / float(r_den)
    return 0 if p + r == 0 else (1 + beta * beta) * p * r / (beta * beta * p + r)


class Evaluator:
    def __init__(self, metric, beta=1):
        self.p_num = 0
        self.p_den = 0
        self.r_num = 0
        self.r_den = 0
        self.metric = metric
        self.beta = beta

    def update(self, document):
        if self.metric == ceafe:
            pn, pd, rn, rd = self.metric(document.clusters, document.gold)
        else:
            pn, pd = self.metric(document.clusters, document.mention_to_gold)
            rn, rd = self.metric(document.gold, document.mention_to_cluster)
        self.p_num += pn
        self.p_den += pd
        self.r_num += rn
        self.r_den += rd

    def get_f1(self):
        return f1(self.p_num, self.p_den, self.r_num, self.r_den, beta=self.beta)

    def get_recall(self):
        return 0 if self.r_num == 0 else self.r_num / float(self.r_den)

    def get_precision(self):
        return 0 if self.p_num == 0 else self.p_num / float(self.p_den)

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()

    def get_counts(self):
        return self.p_num, self.p_den, self.r_num, self.r_den


def evaluate_documents(documents, metric, beta=1):
    evaluator = Evaluator(metric, beta=beta)
    for document in documents:
        evaluator.update(document)
    return evaluator.get_precision(), evaluator.get_recall(), evaluator.get_f1()


def b_cubed(clusters, mention_to_gold):
    num, dem = 0, 0

    for c in clusters:
        if len(c) == 1:
            continue

        gold_counts = Counter()
        correct = 0
        for m in c:
            if m in mention_to_gold:
                #qq = mention_to_gold[m]
                qq = tuple(mention_to_gold[m])
                gold_counts[qq] += 1
        for c2, count in gold_counts.iteritems():
            if len(c2) != 1:
                correct += count * count
        
        num += correct / float(len(c))
        dem += len(c)

    return num, dem

def b_cubed_n(clusters, mention_to_gold):
    num, dem = 0, 0

    for c in clusters:

        gold_counts = Counter()
        correct = 0
        for m in c:
            if m in mention_to_gold:
                #qq = mention_to_gold[m]
                qq = tuple(mention_to_gold[m])
                gold_counts[qq] += 1

        for c2, count in gold_counts.iteritems():
            #if len(c2) != 1:
            correct += count * count
        
        num += correct / float(len(c))
        dem += len(c)

    return num, dem



def muc(clusters, mention_to_gold):
    tp, p = 0, 0
    for c in clusters:
        p += len(c) - 1
        tp += len(c)
        linked = set()
        for m in c:
            if m in mention_to_gold:
                linked.add(mention_to_gold[m])
            else:
                tp -= 1
        tp -= len(linked)
    return tp, p


def phi4(c1, c2):
    return 2 * len([m for m in c1 if m in c2]) / float(len(c1) + len(c2))


def ceafe(clusters, gold_clusters):
    #clusters = [c for c in clusters if len(c) != 1]
    clusters = [c for c in clusters]
    scores = np.zeros((len(gold_clusters), len(clusters)))
    for i in range(len(gold_clusters)):
        for j in range(len(clusters)):
            scores[i, j] = phi4(gold_clusters[i], clusters[j])
    matching = linear_assignment(-scores)
    similarity = sum(scores[matching[:, 0], matching[:, 1]])
    return similarity, len(clusters), similarity, len(gold_clusters)


def lea(clusters, mention_to_gold):
    num, dem = 0, 0

    for c in clusters:
        if len(c) == 1:
            continue

        common_links = 0
        all_links = len(c) * (len(c) - 1) / 2.0
        for i, m in enumerate(c):
            if m in mention_to_gold:
                for m2 in c[i + 1:]:
                    if m2 in mention_to_gold and mention_to_gold[m] == mention_to_gold[m2]:
                        common_links += 1

        num += len(c) * common_links / float(all_links)
        dem += len(c)

    return num, dem

def list2dict(l):
    rd = {}
    for item in l:
        for word in item:
            rd[word] = tuple(item)
    return rd
if __name__ == "__main__":

    while True:
        line = sys.stdin.readline()
        if not line:break
        line = line.strip().split(" ")
        gold = []
        for st in line:
            item = []
            for word in st:
                item.append(word)
            gold.append(item)
        gold_d = list2dict(gold)

        line = sys.stdin.readline()
        line = line.strip().split(" ")
        predict = []
        for st in line:
            item = []
            for word in st:
                item.append(word)
            predict.append(item)
        predict_d = list2dict(predict)

        print "gold:",(gold)
        print "predict:",(predict)

        c,d = muc(gold,predict_d)
        a,b = muc(predict,gold_d)
        print "MUC: precision: %d/%d, %f  recall: %d/%d, %f"%(a,b,float(a)/float(b),c,d,float(c)/float(d))

        c,d = b_cubed_n(gold,predict_d)
        a,b = b_cubed_n(predict,gold_d)
        print "BCUBED: precision: %f/%f, %f  recall: %f/%f, %f"%(a,b,float(a)/float(b),c,d,float(c)/float(d))

        a,b,c,d = ceafe(predict,gold)
        print "CEAF: precision: %f/%f, %f  recall: %f/%f, %f"%(a,b,float(a)/float(b),c,d,float(c)/float(d))

        print

        line = sys.stdin.readline()


    '''

    gold = [[(0,1)],[(3,6),(9,9)],[(11,12),(14,14),(16,18)]]
    gold_d = {}
    gold_d[(0,1)] = ((0,1))
    gold_d[(3,6)] = ((3,6),(9,9))
    gold_d[(9,9)] = ((3,6),(9,9))
    gold_d[(11,12)] = ((11,12),(14,14),(16,18))
    gold_d[(14,14)] = ((11,12),(14,14),(16,18))
    gold_d[(16,18)] = ((11,12),(14,14),(16,18))

    
    predict = [[(0,1)],[(11,12),(14,14)]]
    predict_d = {}
    predict_d[(0,1)]=((0,1))
    predict_d[(11,12)]=((11,12),(14,14))
    predict_d[(14,14)]=((11,12),(14,14))
    
    #predict = [[(0,1)],[(11,12),(14,14)]]
    #print muc(gold,predict_d)
    #a,b = b_cubed(gold,predict_d)
    #a,b = b_cubed(predict,gold_d)
    #print a/b

    pn, pd, rn, rd = ceafe(predict, gold)
    print pn,pd,rn,rd
    '''
