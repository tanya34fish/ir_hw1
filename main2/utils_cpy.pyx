# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import os
import codecs
import math
import operator
import pickle
import time
import heapq
 
'''tested:OK'''
def readDocFile(doc_dir, doclist_file):
    docFilename = readDocFilename(doclist_file);

    #alldoc = os.listdir(doc_dir);
    docInfoList = [];
    docLengthList = [];
    for doc in docFilename.itervalues():
        doc_dict = parseDocXml(doc);
        docInfoList.append(doc_dict);
        docLengthList.append(len(doc_dict['text']));
        print(doc)
    return (docInfoList, docLengthList);

'''tested:OK'''
def parseDocXml(doc_file):
    doc_dict = dict();
    tree = ET.parse(doc_file);
    root = tree.getroot();
    doc_dict['docID'] = root[0][0].text;
    doc_dict['date'] = root[0][1].text;
    doc_dict['title'] = root[0][2].text;
    doc_dict['text'] = '';
    for paragraph in root[0][3]:
        doc_dict['text'] += paragraph.text;

    '''get doc's length'''
    if isinstance(doc_dict['text'], unicode) == False:
        term_list = list(doc_dict['text'].decode('utf-8'));
    else:
        term_list = list(doc_dict['text']);

    '''delete \n '''
    doc_dict['text'] = filter(lambda term: term != u'\n', term_list);

    '''TODO:remove stopword and punctuation(comma,semicolon...)'''

    return doc_dict;

'''tested:OK'''
def parseQueryXml(query_file):
    query_list = [];
    tree = ET.parse(query_file);
    root = tree.getroot();
    for idx in xrange(len(root)):
        query_dict = dict();
        query_dict['topicID'] = root[idx][0].text;
        query_dict['title'] = root[idx][1].text;
        query_dict['question'] = root[idx][2].text;
        query_dict['narrative'] = root[idx][3].text;
        query_dict['concepts'] = root[idx][4].text;
        query_list.append(query_dict);
    return query_list;

def removeStopwords(doc_list):
    pass;
    return;

'''tested:OK'''
def readVocab(vocab_file):
    vocab = {};
    with codecs.open(vocab_file,'r',encoding='utf8') as f:
        f.readline();
        idx = 1;
        for line in f:
            vocab[line.strip()] = idx;
            idx += 1;
    return vocab;

'''tested:OK'''
def readDocFilename(doclist_file):
    docFilename = {};
    with open(doclist_file,'r') as f:
        idx = 0;
        for line in f:
            docFilename[idx] = line.strip();
            idx += 1;
    return docFilename;

"""TODO"""
def evaluateMAP(ans_file, predict_file):
    answer = {};
    with open(ans_file,'r') as f:
        for line in f:
            line = line.strip().split();
            if line[0] not in answer:
                answer[line[0]] = [line[1]];
            else:
                answer[line[0]].append(line[1]);
    
    predict = {};
    with open(predict_file,'r') as f:
        for line in f:
            line = line.strip().split();
            if line[0] not in predict:
                predict[line[0]] = [line[1]];
            else:
                predict[line[0]].append(line[1]);

    map = 0.0;
    #for (key,value) in answer.iteritems():
    #    map += oneSampleMAP(value,predict[key]);
    #map /= float(len(answer));
    for (key,value) in predict.iteritems():
        map += oneSampleMAP(answer[key],value);

    return map;

def oneSampleMAP(answer,predict):
    rightans = 0;
    avep = 0.0;
    for (idx,p) in enumerate(predict):
        if p in answer:
            rightans += 1;
            avep += rightans / (float)(idx+1);
    avep /= len(answer);
    return avep;
    
class informationRetrieval():
    def __init__(self, docLengthList, queryList, vocab):
        self.invertFile = [];
        ##n-gram dictionary {n-gram:id}
        self.nGramDict = {};
        ##length of every document
        self.docLengthList = list(docLengthList);
        ###
        self.docNum = len(self.docLengthList);
        ##score of every document
        self.docScore = {};
        ##
        self.query = list(queryList);
        ##
        self.vocab = vocab;
        ##
        self.doc_tf = {};
        ##id to ngram(for debug)
        self.idToNGram = {};

    '''tested:OK, isolated for debug''' 
    def genNGramDict(self, inverted_file):
        with open(inverted_file,'r') as f:
            idx = 0;
            while True:
                line = f.readline();
                ##eof
                if not line:
                    break;
                            
                info = line.strip().split();
                if len(info) == 3:
                    gram_str = info[0];
                    if int(info[1]) != -1:
                        gram_str += '_' + info[1];
                  
                    self.nGramDict[gram_str] = idx;
                    idx += 1;
        return;

    '''tested:OK'''
    def readInvertedFile(self, inverted_file):
         
        tic = time.time();
        with open(inverted_file,'r') as f:
            idx = 0;
            moveonLine = 0;
            moveonIndex = 0;
            while True:
                line = f.readline();
                ##eof
                if not line:
                    break;
                            
                info = line.strip().split();
                if len(info) == 3:
                    gram_str = info[0];
                    if int(info[1]) != -1:
                        gram_str += '_' + info[1];
                  
                    self.nGramDict[gram_str] = idx;
                    self.idToNGram[idx] = gram_str;
                    idx += 1;
                    moveonLine = int(info[2]);
                    moveonIndex = 0;
                else:
                    if moveonIndex == 0:
                        tmp = [];

                    tmp.append((int(info[0]),int(info[1])));
                    
                    if moveonIndex == moveonLine-1:
                        self.invertFile.append(tmp);
                    moveonIndex += 1;
        #pickle.dump(self.invertFile,open('invertFile.pkl','wb'));
        
        #self.invertFile = pickle.load(open('invertFile.pkl','rb'));
        print(time.time()-tic);
        return;
    
    def termToDocWeight(self, id):
        tfidf = {};
        for (doc_id, freq) in self.invertFile[id]:
            tfidf[doc_id] = self.doc_tf[doc_id][id];
            tfidf[doc_id] *= math.log(self.docNum/ float(len(self.invertFile[id])), 10);
        return tfidf;
    
    '''tested:OK'''
    def termToNGramKey(self, term):
        ##unigram
        if len(term) == 1:
            return str(self.vocab[term[0]]);
        else:
            delimit = '_';
            term_convert = [str(self.vocab[tmp]) for tmp in term];
            return delimit.join(term_convert);
    
    '''tested:OK'''
    def queryToNGram(self, query, minN, maxN):
        """(minN, maxN) = (1, 2) for unigram and bigram"""
        """return nGramList {ngram_idx:term frequency}"""
         
        ##query to term list
        if isinstance(query, unicode) == False:
            term_list = list(query.decode('utf-8'));
        else:
            term_list = list(query);

        n_gram = len(term_list);
        nGramList = {};
        for i in xrange(n_gram):
            for j in xrange(i+minN, min(n_gram, i+maxN)+1):
                term = term_list[i:j];
                ##if vocab in term is not found in self.vocab
                notFound = False;
                for k in term:
                    if k not in self.vocab:
                        notFound = True;
                        break;
                if notFound:
                    continue;

                ngram_key = self.termToNGramKey(term);
                if ngram_key in self.nGramDict:
                    ngram_idx = self.nGramDict[ngram_key];
                    if ngram_idx in nGramList:
                        nGramList[ngram_idx] += 1;
                    else:
                        nGramList[ngram_idx] = 1;

        
        for ngram_idx in nGramList.iterkeys():
            #print(self.idToNGram[ngram_idx]);
            nGramList[ngram_idx] /= float(n_gram);
            nGramList[ngram_idx] *= math.log(self.docNum/ float(len(self.invertFile[ngram_idx])), 10);
        return nGramList;
    
    
    def calDocTf(self):
        for (term_idx,tmp_list) in enumerate(self.invertFile):
            for (doc_id, term_freq) in tmp_list:
                if doc_id in self.doc_tf:
                    self.doc_tf[doc_id][term_idx] = float(term_freq)/float(self.docLengthList[doc_id]);
                else:
                    self.doc_tf[doc_id] = {};
                    self.doc_tf[doc_id][term_idx] = float(term_freq)/float(self.docLengthList[doc_id]);
        
        return;
    
    def calScore(self, nGramList, k):
        tic = time.time();
        ##initialize self.docScore;
        for idx in range(self.docNum):
            self.docScore[idx] = 0.0;
        
        denom_query = 0.0;
        for (ngram_idx, q_tfidf) in nGramList.iteritems():
            denom_query += q_tfidf ** 2;
            w_term_doc = self.termToDocWeight(ngram_idx);
            for (doc_id, w_tfidf) in w_term_doc.iteritems():
                self.docScore[doc_id] += q_tfidf * w_tfidf;
        
        ##cosine similarity
        for doc_id in range(len(self.docScore)):
            if (self.docScore[doc_id] == 0.0) or (self.docLengthList[doc_id] == 0):
                continue;
            else:
                denom_doc = 0.0;
                for tfidf in self.doc_tf[doc_id].itervalues():
                    denom_doc += tfidf ** 2;
                self.docScore[doc_id] /= math.sqrt(denom_query * denom_doc);
        
        new_rel = heapq.nlargest(k, self.docScore, key=lambda j: self.docScore[j]);
        new_nonrel = heapq.nsmallest(k, self.docScore, key=lambda j: self.docScore[j]);
        print('took %g secs done calscore...' %(time.time()-tic));
        return (new_rel, new_nonrel)
    
             
    def assignZero(self, qweights):
        for idx in range(len(self.nGramDict)):
            qweights[idx] = 0.0;
        return;
    
    def rocchio(self, docFilename, query_topic, query, output, k=100, iter=50, alpha=1.0, beta=0.8, gamma=0.1, relFeed=True):

        print('use cython...')
        """first time"""
        nGramList = self.queryToNGram(query, 1, 2);
        (rel,nonrel) = self.calScore(nGramList, k);
        if relFeed == False:
            outputFile = open(output,'w');
            for doc_idx in rel:
                outputFile.write('%s %s\n' %(query_topic, docFilename[doc_idx].lower()[-15:]));
            return;
        
        for iter_idx in range(iter):
            tic = time.time();
            relDocs = {};
            nonrelDocs = {};
            doc_rel_bool = [0] * self.docNum;
            #for doc_id in rel:
            #    doc_rel_bool[doc_id] = 1;
            #for doc_id in nonrel:
            #    doc_rel_bool[doc_id] = -1;
            tic2 = time.time();
            for doc_id in rel:
                doc_rel_bool[doc_id] = 1;
                ##filter zero words
                if doc_id not in self.doc_tf:
                    continue;
                for (term,freq) in self.doc_tf[doc_id].iteritems():
                    if term in relDocs:
                        relDocs[term] += freq;
                    else:
                        relDocs[term] = freq;

            for doc_id in nonrel:
                doc_rel_bool[doc_id] = -1;
                ##filter zero words
                if doc_id not in self.doc_tf:
                    continue;
                for (term,freq) in self.doc_tf[doc_id].iteritems():
                    if term in nonrelDocs:
                        nonrelDocs[term] += freq;
                    else:
                        nonrelDocs[term] = freq;
            qweights = {};
            self.assignZero(qweights);
            for (term_idx,tmp_list) in enumerate(self.invertFile):
                term_idf = math.log(float(self.docNum)/float(len(tmp_list)), 10);
                for (doc_id,freq) in tmp_list:
                    if doc_rel_bool[doc_id] == 1:
                        qweights[term_idx] += beta * term_idf * (relDocs[term_idx] / len(rel));
                    elif doc_rel_bool[doc_id] == -1:
                        qweights[term_idx] -= gamma * term_idf * (nonrelDocs[term_idx] / len(nonrel));
            #qweights = {};
            #self.assignZero(qweights);
            #tic2 = time.time();
            #for (idx, doc_bool) in enumerate(doc_rel_bool):
            #    ##filter zero words
            #    if idx not in self.doc_tf:
            #        continue;
            #    #rel
            #    if doc_bool == 1:
            #        for (key,value) in self.doc_tf[idx].iteritems():
            #            idf = math.log(float(self.docNum)/float(len(self.invertFile[key])),10);
            #            qweights[key] += beta * idf * value / float(len(rel));
            #    #non-rel
            #    #else:
            #    #    for (key,value) in self.doc_tf[idx].iteritems():
            #    #        idf = math.log(float(self.docNum)/float(len(self.invertFile[key])),10);
            #    #        qweights[key] -= gamma * idf * value / float(len(nonrel));

            print('took %g secs for cal qweights...' %(time.time()-tic2));
            
            for (term_idx,q_tfidf) in nGramList.iteritems():
                qweights[term_idx] += alpha * q_tfidf;

            """negative weights become zero"""
            for (term_idx,weight) in qweights.iteritems():
                if weight < 0.0:
                    qweights[term_idx] = 0.0;

            new_weights = dict((k,v) for k,v in qweights.iteritems() if v!=0.0);
            """query again""" 
            (rel,nonrel) = self.calScore(new_weights, k);
            
            print('Iteration %d: took %g secs' %(iter_idx,time.time()-tic));
            
        outputFile = open(output,'w');
        for doc_idx in rel:
            outputFile.write('%s %s\n' %(query_topic, docFilename[doc_idx].lower()[-15:]));

        return;
