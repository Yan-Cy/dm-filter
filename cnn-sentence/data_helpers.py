import numpy as np
import re
import itertools
from collections import Counter
import gensim
from segChinese import filterChinese

'''
class MySentences(object):
    def __init__(self, filename, bigram, trigram):
        self.filename = filename
        self.bigram = bigram
        self.trigram = trigram
                  
    def __iter__(self):
        with open(self.filename) as f:
            for line in f:
                yield ' '.join(self.trigram[self.bigram[filterChinese(unicode(line, 'utf-8'))]])

def load_sentences(filename):
    bigram = gensim.models.phrases.Phrases.load('../word2vec/models/bi.new')
    trigram = gensim.models.phrases.Phrases.load('../word2vec/models/tri.new')
    sentences = MySentences(filename, bigram, trigram)
    return sentences
'''

def build_vocabulary(sentences):
    model = gensim.models.Word2Vec.load('../word2vec/models/dm_model.1word')
    #model = gensim.models.Word2Vec(sentences, size=100, window=5, min_count=5, workers=4) 
    #print 'Vocabulary Size:', len(model.wv.syn0)
    max_document_length = 30

    x = np.zeros((len(sentences), max_document_length))
    for i, sentence in enumerate(sentences):
        for j, word in enumerate(sentence):
            if j >= max_document_length:
                break
            if word in model.wv.vocab:
                x[i,j] = model.wv.vocab[word].index

    return x, model.wv.syn0

def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples

    #bigram = gensim.models.phrases.Phrases.load('../word2vec/models/bi.new')
    #trigram = gensim.models.phrases.Phrases.load('../word2vec/models/tri.new')
    #x_text = [trigram[bigram[filterChinese(unicode(sent, 'utf-8'))]]  for sent in x_text]
    
    x_text = [filterChinese(unicode(sent, 'utf-8')) for sent in x_text]

    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
