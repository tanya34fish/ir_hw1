import argparse
import os
import pickle
from multiprocessing import Process
import utils_cpy
import time

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
        relFeed = True;
    else:
        relFeed = False;

    query_file = opts.i[0];
    ranked_list = opts.o[0];
    model_dir = opts.m[0];
    doc_dir = opts.d[0];
    
    vocab_file = model_dir + '/vocab.all';
    doclist_file = model_dir + '/file-list';
    invert_file = model_dir + '/inverted-file';
    

    queryList = utils_cpy.parseQueryXml(query_file);
    vocab = utils_cpy.readVocab(vocab_file);
    #(docInfoList, docLengthList) = utils_cpy.readDocFile(doc_dir, doclist_file);
    #pickle.dump(docLengthList, open('docLengthList.pkl', 'wb')); 
    docLengthList = pickle.load(open('../docLengthList.pkl', 'rb'));
    
    ir = utils_cpy.informationRetrieval(docLengthList, queryList, vocab);
    ir.readInvertedFile(invert_file);
    #cal term frequency 
    ir.calDocTf(); 
 
    topK = 100;
    docFilename = utils_cpy.readDocFilename(doclist_file);

    #query_topic = queryList[0]['topicID'][-3:];
    #ir.rocchio(docFilename, query_topic, queryList[0]['concepts'], 'output/0.predict', relFeed=False);
    p = [];
    for i in range(len(queryList)):
        query_topic = queryList[i]['topicID'][-3:];
        p.append(Process(target=ir.rocchio, args= (docFilename, query_topic, queryList[i]['narrative'], ranked_list+'/%d.predict' % i)))
    for i in range(len(queryList)):
        p[i].start();
    
    #merge = open('output_test/merge.predict','w');
    #for i in range(20):
    #    with open('output_test/%d.predict' %(i), 'r') as f:
    #        for line in f:
    #            line = line.strip().split();
    #            merge.write('%s %s\n' %(line[0],line[1]));
    #merge.close();

    #print(utils_cpy.evaluateMAP('../query/ans-train', 'output_train6.predict'));
    return;

if __name__ == "__main__":
    main();
