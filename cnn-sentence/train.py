#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import pickle

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "../danmu/positive.train", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "../danmu/negative.train", "Data source for the negative data.")
#tf.flags.DEFINE_string('pretrained_model', './runs/1492488399/checkpoints/model-17000', 'Used to resume training')

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 256, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 200, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 200, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

'''
# Data Preparation
# ==================================================

# Load data
print("Loading data...")

#x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
#pickle.dump(x_text, open('mid_data/x_text.data', 'wb'))
#np.save('mid_data/y.data', y)

x_text = pickle.load( open('mid_data/x_text.data', 'rb') )
y = np.load('mid_data/y.data.npy')


# Build vocabulary
print 'Building vocabulary'
#max_document_length = max([len(x.split(" ")) for x in x_text])
#vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
#vocab_processor = learn.preprocessing.VocabularyProcessor,restore('vocabulary.processor')
#vocab_vector = numpy.load('vocabulary.vector')
#x = np.array(list(vocab_processor.fit_transform(x_text)))
#x = np.array(list(vocab_processor.transform(x_text)))
x, vocab_vector = data_helpers.build_vocabulary(x_text)

np.save('mid_data/x.data', x)
np.save('mid_data/vocab_vector.data', vocab_vector)
#x = np.load('mid_data/x.data.npy')
#vocab_vector = np.load('mid_data/vocab_vector.data.npy')

#del x_text

# Randomly shuffle data
print 'Shuffling data'
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/test set
# TODO: This is very crude, should use cross-validation
print 'Spliting data'
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
print("Vocabulary Size: {:d}".format(len(vocab_vector)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

del x
del y
del x_shuffled
del y_shuffled

np.save('mid_data/x_train.data', x_train)
np.save('mid_data/x_dev.data', x_dev)
np.save('mid_data/y_train.data', y_train)
np.save('mid_data/y_dev.data', y_dev)

'''
vocab_vector = np.load('mid_data/vocab_vector.data.npy')
x_train = np.load('mid_data/x_train.data.npy')
x_dev = np.load('mid_data/x_dev.data.npy')[:100000]
y_train = np.load('mid_data/y_train.data.npy')
y_dev = np.load('mid_data/y_dev.data.npy')[:100000]

# Training
# ==================================================

print 'Begin Training'

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement,
      device_count={'CPU': 20})
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=x_train.shape[1],
            num_classes=y_train.shape[1],
            #vocab_size=len(vocab_processor.vocabulary_),
            vocab_size=len(vocab_vector),
            embedding_size=FLAGS.embedding_dim,
            vocab_vector = vocab_vector,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
        #saver.restore(sess, FLAGS.pretrained_model)

        # Write vocabulary
        #vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            #print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)
            return [accuracy, loss]

        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        dev_record = open('dev.record', 'w')
        dev_record.write('Step      Accuracy      Loss\n')
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                print 'hahahahah'
                sum_accuracy = 0.0
                sum_loss = 0.0
                cnt = 0.0
                num_batches = 200
                for dev_num in range(num_batches):
                    s = dev_num * 500
                    t = s + 500 
                    x_dev_batch = x_dev[s:t]
                    y_dev_batch = y_dev[s:t]
                    #accuracy += dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    batch_accuracy, batch_loss = dev_step(x_dev_batch, y_dev_batch, writer=dev_summary_writer)
                    sum_accuracy += batch_accuracy
                    sum_loss += batch_loss
                    cnt += 1
                print 'Average Accuracy:', sum_accuracy / cnt 
                print 'Average Loss:', sum_loss / cnt
                dev_record.write(str(current_step) + '  ' + str(sum_accuracy / cnt) + ' ' + str(sum_loss / cnt) + '\n') 
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))

        dev_record.close()

