import os
import numpy as np
from segChinese import filterChinese
import gensim
from datetime import datetime

class MySentences(object):
    def __init__(self, filename):
        self.filename = filename
                 
    def __iter__(self):
        with open(self.filename) as f:
            for line in f:
                yield filterChinese(unicode(line.split('\t')[5], 'utf-8'))


def get_sentences(dmfile):
    with open(dmfile) as f:
        sentences = [filterChinese(unicode(line, 'utf-8')) for line in f.readlines()]
    #with open(dmfile) as f:
    #    sentences = [filterChinese(unicode(line.split('\t')[5], 'utf-8')) for line in f.readlines()]
    return sentences


def train_vec(sentences):
    '''
    #bigram = gensim.models.phrases.Phrases(sentences)
    bigram = gensim.models.phrases.Phrases.load('models/bi.phrases')
    bigram.add_vocab(sentences)
    bigram.save('models/bi.new')
    print 'bigram phrases done', str(datetime.now())
    
    #trigram = gensim.models.phrases.Phrases(bigram[sentences])
    trigram = gensim.models.phrases.Phrases.load('models/tri.phrases')
    trigram.add_vocab(bigram[sentences])
    trigram.save('models/tri.new')
    print 'trigram phrases done', str(datetime.now())
    
    #model = gensim.models.Word2Vec(transformer[trigram[bigram[sentences]]], size=100, window=5, min_count=5, workers=10)
    model = gensim.models.Word2Vec.load('models/dm.model')
    model.train(trigram[bigram[sentences]])
    print 'train model done', str(datetime.now())
    
    modelfile = 'models/dm_model.new'
    model.save(modelfile)
    print 'save model done', str(datetime.now())
    '''

    #modelfile = 'models/dm_model.new'
    #model = gensim.models.Word2Vec.load(modelfile)
    model = gensim.models.Word2Vec(sentences, size=100, window=5, min_count=5, workers=10)
    model.save('models/dm_model.1word')


if __name__ == '__main__':
    dmfile = 'models/src_dm.txt'
    #sentences = MySentences(dmfile)
    sentences = get_sentences(dmfile)
    #with open('dm.sentence', 'w') as sfile:
    #    for sentence in sentences:
    #        sfile.write((' '.join(sentence) + '\n').encode('UTF-8'))
    print 'loaded sentences', str(datetime.now())
   
    train_vec(sentences)
