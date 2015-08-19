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
    for (key,value) in answer.iteritems():
        map += oneSampleMAP(value,predict[key]);
    map /= float(len(answer));
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
    
    def removeStopwords(stopword_list):
        pass; 
        return;

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
        for i in xrange(n_gram-1):
            #for j in xrange(i+minN, min(n_gram, i+maxN)+1):
            term = term_list[i:i+2];
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
                #if ngram_idx in nGramList:
                nGramList[ngram_idx] = 0;
                #else:
                #    nGramList[ngram_idx] = 1;

        max_tf = max([v for (u,v) in nGramList.iteritems()]);
        for ngram_idx in nGramList.iterkeys():
            #print(self.idToNGram[ngram_idx]);
            #nGramList[ngram_idx] = math.log(float(n_gram);
            #nGramList[ngram_idx] = 0.5 + 0.5 * nGramList[ngram_idx] / float(max_tf);
            nGramList[ngram_idx] = math.log(1.0 + self.docNum/ float(len(self.invertFile[ngram_idx])), 2);
        return nGramList;
    
    
    def calDocTf(self):
        for (term_idx,tmp_list) in enumerate(self.invertFile):
            for (doc_id, term_freq) in tmp_list:
                if doc_id in self.doc_tf:
                    #self.doc_tf[doc_id][term_idx] = float(term_freq)/float(self.docLengthList[doc_id]);
                    self.doc_tf[doc_id][term_idx] = 1.0 + math.log(float(term_freq),2);
                else:
                    self.doc_tf[doc_id] = {};
                    #self.doc_tf[doc_id][term_idx] = float(term_freq)/float(self.docLengthList[doc_id]);
                    #self.doc_tf[doc_id][term_idx] = float(term_freq);
                    self.doc_tf[doc_id][term_idx] = 1.0 + math.log(float(term_freq),2);
        
        return;
    
    def calScore(self, nGramList, k):
        tic = time.time();
        ##initialize self.docScore;
        for idx in range(self.docNum):
            self.docScore[idx] = 0.0;
        
        denom_query = 0.0;
        for (ngram_idx, q_tfidf) in nGramList.iteritems():
            denom_query += q_tfidf ** 2;
        
        ##cosine similarity
        for doc_id in range(self.docNum):
            denom_doc = 0.0;
            if self.docLengthList[doc_id] == 0:
                continue;
            for (term_idx, tf) in self.doc_tf[doc_id].iteritems():
                doc_tfidf = tf; #* math.log(self.docNum/ float(len(self.invertFile[term_idx])), 2);
                denom_doc += doc_tfidf ** 2;
                if term_idx in nGramList:
                    self.docScore[doc_id] += nGramList[term_idx] * doc_tfidf;
            self.docScore[doc_id] /= math.sqrt(denom_query * denom_doc);
        
        new_rel = heapq.nlargest(k, self.docScore, key=lambda j: self.docScore[j]);
        new_nonrel = heapq.nsmallest(k, self.docScore, key=lambda j: self.docScore[j]);
        print('took %g secs done calscore...' %(time.time()-tic));
        return (new_rel, new_nonrel)
    
             
    def assignWeights(self, qweights, nGramList):
        for idx in range(len(self.nGramDict)):
            qweights[idx] = 0.0;
        for (term_idx,q_tfidf) in nGramList.iteritems():
            qweights[term_idx] = q_tfidf;
        return;
    
    def rocchio(self, docFilename, query_topic, query, output, k=100, iter=2, alpha=1.0, beta=0.75, gamma=0.15, relFeed=True):

        print('use cython...')
        """first time"""
        nGramList = self.queryToNGram(query, 1, 2);
        (rel,nonrel) = self.calScore(nGramList, k);
        if relFeed == False:
            outputFile = open(output,'w');
            for doc_idx in rel:
                outputFile.write('%s %s\n' %(query_topic, docFilename[doc_idx].lower()[-15:]));
            return;
        
        qweights = {};
        self.assignWeights(qweights, nGramList);

        for iter_idx in range(iter):
            tic = time.time();
            doc_rel_bool = [0] * self.docNum;
            rel = list(rel[:10])
            nonrel = list(nonrel[:10])
            for doc_id in rel:
                doc_rel_bool[doc_id] = 1;
            for doc_id in nonrel:
                doc_rel_bool[doc_id] = -1;
            
            for term_idx in qweights.iterkeys():
                qweights[term_idx] *= alpha;
            
            tic2 = time.time();
            for (idx, doc_bool) in enumerate(doc_rel_bool):
                ##filter zero words
                if idx not in self.doc_tf:
                    continue;
                #rel
                if doc_bool == 1:
                    for (key,value) in self.doc_tf[idx].iteritems():
                        #doc_tfidf = value * math.log(float(self.docNum)/float(len(self.invertFile[key])),2);
                        doc_tfidf = value;# * math.log(float(self.docNum)/float(len(self.invertFile[key])),2);
                        qweights[key] += beta * doc_tfidf / float(len(rel));
                #non-rel
                elif doc_bool == -1:
                    for (key,value) in self.doc_tf[idx].iteritems():
                        #doc_tfidf = value * math.log(float(self.docNum)/float(len(self.invertFile[key])),2);
                        doc_tfidf = value;# * math.log(float(self.docNum)/float(len(self.invertFile[key])),2);
                        qweights[key] -= gamma * doc_tfidf / float(len(nonrel));

            print('took %g secs for cal qweights...' %(time.time()-tic2));
            

            """negative weights become zero"""
            #for (term_idx,weight) in qweights.iteritems():
            #    if weight < 0.0:
            #        qweights[term_idx] = 0.0;

            #new_weights = dict((k,v) for k,v in qweights.iteritems() if v!=0.0);
            """query again""" 
            (rel,nonrel) = self.calScore(qweights, k);
            
            print('Iteration %d: took %g secs' %(iter_idx,time.time()-tic));
            
        outputFile = open(output,'w');
        for doc_idx in rel:
            outputFile.write('%s %s\n' %(query_topic, docFilename[doc_idx].lower()[-15:]));

        return;
