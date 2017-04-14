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
        sentences = [filterChinese(unicode(line.split('\t')[5], 'utf-8')) for line in f.readlines()]
    return sentences

def gen_vec(sentences):
    #bigram = gensim.models.phrases.Phrases(sentences)
    #bigram.save('models/bi.phrases')
    bigram = gensim.models.phrases.Phrases.load('models/bi.phrases')
    print 'bigram phrases done', str(datetime.now())
    
    #trigram = gensim.models.phrases.Phrases(bigram[sentences])
    #trigram.save('models/tri.phrases')
    trigram = gensim.models.phrases.Phrases.load('models/tri.phrases')
    print 'trigram phrases done', str(datetime.now())
    
    #transformer = gensim.models.phrases.Phrases(trigram[bigram[sentences]])
    #transformer.save('models/quod.phrases')
    transformer = gensim.models.phrases.Phrases.load('models/quod.phrases')
    print 'quodgram phrases done', str(datetime.now())
    
    #phrases = transformer.export_phrases(trigram[bigram[sentences]])
    #with open('models/phrases.txt', 'w') as f:
    #    for phrase, score in phrases:
    #        f.write('{0}   {1}\n'.format(phrase, score))

    count = 0
    t = []
    for sentence in sentences:
        t.append(sentence)
        count += 1
        if count == 5:
            break

    print t
    corpus =  transformer[trigram[bigram[t]]]
    for sentence in corpus:
        print sentence

    #model = gensim.models.Word2Vec(transformer[trigram[bigram[sentences]]], size=100, window=5, min_count=5, workers=10)
    #print 'train model done', str(datetime.now())
    
    #modelfile = 'dm.model'
    #model.save(modelfile)
    #print 'save model done', str(datetime.now())
    
    #vectorfile = 'dm.vector'
    #np.save(vectorfile, model.wv)
    #print 'save vector done', str(datetime.now())


if __name__ == '__main__':
    dmfile = '../scripts/dm_100m.txt'
    sentences = MySentences(dmfile)
    #sentences = get_sentences(dmfile)
    #with open('dm.sentence', 'w') as sfile:
    #    for sentence in sentences:
    #        sfile.write((' '.join(sentence) + '\n').encode('UTF-8'))
    print 'loaded sentences', str(datetime.now())
   
    gen_vec(sentences)

