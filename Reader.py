#coding=utf8
import sys
import Mention
import json
import timeit

def read_from_file(fil,goldf,language):
    line_num = len(open(goldf).readlines())
    
    total_docs = []
    
    print >> sys.stderr, "Read from %s and %s documents"%(fil,goldf)

    f = open(fil)
    gold_file = open(goldf)

    total_start_time = timeit.default_timer()
    i = 0
    while True:
        line = f.readline()
        if not line:break
        i += 1

        if i >= 5:
            break 
        
        start_time = timeit.default_timer()
        line = line.strip()
        s = json.loads(line)
        line = gold_file.readline()
        js = json.loads(line)
        gold_chain = js[js.keys()[0]]

        document = Mention.Document(s,gold_chain,language)
        #total_docs.append(document)
        yield document

        end_time = timeit.default_timer()
        print >> sys.stderr, "Done %d/%d document - %.3f seconds"%(i,line_num,end_time-start_time)

    total_end_time = timeit.default_timer()
    print >> sys.stderr, "Total use %.3f seconds"%(total_end_time-total_start_time)
    #return total_docs
