import tensorflow as tf
import numpy as np
from tensorflow.contrib import learn
import data_helpers
import gensim
import pickle
from datetime import datetime

'''
positive_data_file = '../danmu/positive.train'
negative_data_file = '../danmu/negative.train'
#positive_data_file = '../danmu/positive.10'
#negative_data_file = '../danmu/negative.10'
datafile = '../danmu/all_dm.txt'
#datafile = '../danmu/positive.10'

print 'Init done', str(datetime.now())

#x_text = data_helpers.load_sentences(datafile)
x_text, y = data_helpers.load_data_and_labels(positive_data_file, negative_data_file)
print 'Load data done', str(datetime.now())

max_document_length = max([len(x.split(" ")) for x in x_text])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))
vocab_processor.save('vocabulary.processor')
print 'Build vocab done.', str(datetime.now())


vocab_dict = vocab_processor.vocabulary_._mapping
sorted_vocab = sorted(vocab_dict.items(), key = lambda x : x[1])
vocabulary = list(list(zip(*sorted_vocab))[0])
print 'Sort vocabulary done', str(datetime.now())
#with open('vocabulary.sorted', 'w') as f:
#    for vocab in vocaulary:
#        f.write(vocab.encode('utf8') + '\n')

with open('vocabulary.sorted', 'wb') as fp:
    pickle.dump(vocabulary, fp)
'''

with open('vocabulary.sorted', 'rb') as fp:
    vocabulary = pickle.load(fp)

#print 'Sort vocab done.'
#print x
#print '----------------------'
#print sorted_vocab
#print '----------------------'
#print vocabulary

model = gensim.models.Word2Vec.load('../word2vec/models/dm_model.1word')
print 'Load model done.', str(datetime.now())

print model.wv.syn0.keys()

vocab_vector = np.zeros([len(vocabulary), 100])
for idx, vocab in enumerate(vocabulary):
    print idx, vocab
    if idx == 0:
        continue
    vocab_vector[idx,:] = model.wv[vocab]
print 'Extract feature done.', str(datetime.now())


np.save('vocabulary.vector', vocab_vector)
print 'Save data done.', str(datetime.now())


