import argparse
import os
import pickle
from multiprocessing import Process
import utils_cpy
import time
import errno
import shutil

def main():
    ###parse arguments
    parser = argparse.ArgumentParser();
    parser.add_argument('-r', action="store_true");
    parser.add_argument('-i',nargs=1);
    parser.add_argument('-o',nargs=1);
    parser.add_argument('-m',nargs=1);
    parser.add_argument('-d',nargs=1);
    opts = parser.parse_args();
    
    if opts.r:
        relFeed_bool = True;
    else:
        relFeed_bool = False;

    query_file = opts.i[0];
    ranked_list = opts.o[0];
    model_dir = opts.m[0];
    doc_dir = opts.d[0];
    
    vocab_file = model_dir + '/vocab.all';
    doclist_file = model_dir + '/file-list';
    invert_file = model_dir + '/inverted-file';
    

    queryList = utils_cpy.parseQueryXml(query_file);
    num_q = len(queryList);
    vocab = utils_cpy.readVocab(vocab_file);
    #print 'load docLengthList..'
    #(docInfoList, docLengthList) = utils_cpy.readDocFile(doc_dir, doclist_file);
    #print 'Done.'
    ##pickle.dump(docLengthList, open('docLengthList.pkl', 'wb')); 
    #docLengthList = pickle.load(open('../docLengthList.pkl', 'rb'));
    docFilename = utils_cpy.readDocFilename(doclist_file);
    
    ir = utils_cpy.informationRetrieval(len(docFilename), queryList, vocab);
    ir.readInvertedFile(invert_file);
    #cal term frequency 
    ir.calDocTf(); 
 
    topK = 100;

    try:
        os.makedirs('tmp');
    except OSError as exc: 
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass;
    p = [];
    for i in range(num_q):
        query_topic = queryList[i]['topicID'][-3:];
        p.append(Process(target=ir.rocchio, args= (docFilename, query_topic, queryList[i]['concepts'], 'tmp/%d.predict' % i, topK, 2, 1.0, 0.75, 0.15, relFeed_bool)))
    for i in range(num_q):
        p[i].start();
   
    #wait until all processes done
    for i in p:
        i.join();
       
    #write all to one file
    merge = open(ranked_list,'w');
    for i in range(num_q):
        with open('tmp/%d.predict' %(i),'r') as f:
            for line in f:
                line = line.strip().split();
                merge.write('%s %s\n' %(line[0],line[1]));
    merge.close();
    #shutil.rmtree('/tmp');
    #print(utils_cpy.evaluateMAP('query/ans-train', 'output_cython/merge.predict'));
    return;

if __name__ == "__main__":
    main();
