import os
import numpy as np
import tensorflow as tf
import data_helpers
from segChinese import filterChinese

class CNN_Sentence(object):

    def __init__(self,
                batch_size = 256,
                checkpoint_dir = './runs/1493697887/checkpoints/',
                allow_soft_placement = True,
                log_device_placement = False
                ):
        self.batch_size = 256

        #print checkpoint_dir
        checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(
                    allow_soft_placement = allow_soft_placement,
                    log_device_placement = log_device_placement)
            self.sess = tf.Session(config=session_conf)
            with self.sess.as_default():
                self.saver = tf.train.import_meta_graph('{}.meta'.format(checkpoint_file))
                self.saver.restore(self.sess, checkpoint_file)
                self.input_x = graph.get_operation_by_name('input_x').outputs[0]
                self.dropout_keep_prob = graph.get_operation_by_name('dropout_keep_prob').outputs[0]
                self.predictions = graph.get_operation_by_name('output/predictions').outputs[0]
                self.scores = graph.get_operation_by_name('output/scores').outputs[0]

    def predict(self, x_text, print_info = False):
        #x_raw = [filterChinese(unicode(sent, 'utf-8')) for sent in x_text]

        with self.sess.as_default():
            x_test, vocab_vector = data_helpers.build_vocabulary(x_raw)
            batches = data_helpers.batch_iter(list(x_test), self.batch_size, 1, shuffle=False)

            num_sentence = len(x_test)
            if print_info:
                count = 0
                total = int((num_sentence-1) / self.batch_size) + 1
                print 'Batch Size: ', self.batch_size
                print 'Num Batches: ', total

            #all_predictions = []
            all_scores = np.zeros((num_sentence,2))
            
            for i, x_test_batch in enumerate(batches):
                if print_info:
                    count += 1
                    if count % 100 == 1:
                        print count, ' / ', total
                batch_scores = self.sess.run(self.scores, {self.input_x: x_test_batch, self.dropout_keep_prob: 1.0})
                #all_predictions = np.concatenate([all_predictions, batch_predictions])
                #print batch_scores
                s = i * self.batch_size
                all_scores[s:s+self.batch_size,:] = batch_scores 
            return all_scores
        
    def evaluate(self, y_test, scores):
        all_predictions = np.argmax(scores, 1)
        correct_predictions = float(sum(all_predictions == y_test))
        accuracy = correct_predictions/float(len(y_test))
        print("Total number of test examples: {}".format(len(y_test)))
       
        print ''
        print 'Confusion Matrix:'
        matrix = np.zeros([2,2])
        for idx, predict in enumerate(all_predictions):
            matrix[1-int(y_test[idx])][1-int(predict)] += 1
        print 'Truth / Predicted       Positive DM      Negative DM'
        print 'Positive DM:              {}             {}'.format(str(matrix[0,0]), str(matrix[0,1]))
        print 'Negative DM:              {}             {}'.format(str(matrix[1,0]), str(matrix[1,1]))
        print ''

        recall = matrix[1,1] / (matrix[1,1] + matrix[1,0])
        precision = matrix[1,1] / (matrix[1,1] + matrix[0,1])
        print("Accuracy :    {:g}".format(accuracy))
        print("Recall   :    {:g}".format(recall))
        print("Precision:    {:g}".format(precision))
        return accuracy


if __name__ == '__main__':
    #posfile = 'data/positive.test'
    #negfile = 'data/negative.test'
    posfile = '../danmu/quanzhi/quanzhi6_1w.pos'
    negfile = '../danmu/quanzhi/quanzhi6.neg'
    x_raw, y_test = data_helpers.load_data_and_labels(posfile, negfile)
    y_test = np.argmax(y_test, axis=1)
    cnn_sentence = CNN_Sentence()
    scores = cnn_sentence.predict(x_raw, print_info = False)
    #print predictions
    #print y_test
    #print len(scores), scores[:20]
    accuracy = cnn_sentence.evaluate(y_test, scores)


