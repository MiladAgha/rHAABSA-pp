#!/usr/bin/env python
# encoding: utf-8

from sklearn.metrics import precision_score, recall_score, f1_score
from utils import load_w2v, batch_index, load_inputs_twitter
from layer_lcr_rot import lcr_rot_hh
from config import *
import numpy as np

def train(train_path):
    
    learning_rate = FLAGS.learning_rate
    keep_prob = FLAGS.keep_prob
    momentum = FLAGS.momentum
    
    tf.reset_default_graph()

    # Load your word embeddings
    word_id_mapping, w2v = load_w2v(FLAGS.train_embedding_path, FLAGS.embedding_dim)
    word_embedding = tf.constant(w2v, name='word_embedding')

    tr_x, tr_sen_len, tr_x_bw, tr_sen_len_bw, tr_y, tr_target_word, tr_tar_len, _, _, _ = load_inputs_twitter(
        train_path, word_id_mapping, FLAGS.max_sentence_len, 'TC', True, FLAGS.max_target_len)
    
    # Create placeholders
    with tf.name_scope('inputs'):
        y = tf.placeholder(tf.float32, [None, FLAGS.n_class])

        x = tf.placeholder(tf.int32, [None, FLAGS.max_sentence_len])
        sen_len = tf.placeholder(tf.int32, None)

        x_bw = tf.placeholder(tf.int32, [None, FLAGS.max_sentence_len])
        sen_len_bw = tf.placeholder(tf.int32, [None])

        target_words = tf.placeholder(tf.int32, [None, FLAGS.max_target_len])
        tar_len = tf.placeholder(tf.int32, [None])

        kp = tf.placeholder(tf.float32)

    # Embedding lookup
    inputs_fw = tf.nn.embedding_lookup(word_embedding, x)
    inputs_bw = tf.nn.embedding_lookup(word_embedding, x_bw)
    target = tf.nn.embedding_lookup(word_embedding, target_words)

    # Define model
    prob = lcr_rot_hh(inputs_fw, inputs_bw, sen_len, sen_len_bw, target, tar_len, kp)

    # Define loss and accuracy
    loss = loss_func(y, prob)
    acc_num, acc_prob = acc_func(y, prob)

    # Create a global step variable
    global_step = tf.Variable(0, name='tr_global_step', trainable=False)

    # Define optimizer
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,  momentum=momentum).minimize(loss, global_step=global_step)

    # Initialize TensorFlow session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        def get_batch_data(x_f, sen_len_f, x_b, sen_len_b, yi, target, tl, batch_size, keep_prob, is_shuffle=True):
            for index in batch_index(len(yi), batch_size, 1, is_shuffle):
                feed_dict = {
                    x: x_f[index],
                    x_bw: x_b[index],
                    y: yi[index],
                    sen_len: sen_len_f[index],
                    sen_len_bw: sen_len_b[index],
                    target_words: target[index],
                    tar_len: tl[index],
                    kp: keep_prob
                }
                yield feed_dict, len(index)

        for epoch in range(FLAGS.n_iter):
            batch_n = 0
            for batch_feed_dict, batch_len in get_batch_data(tr_x, tr_sen_len, tr_x_bw, tr_sen_len_bw, tr_y, tr_target_word, tr_tar_len, FLAGS.batch_size, keep_prob):
                batch_n += 1
                # Run a training step
                _, current_loss, current_acc_num, current_acc_prob = sess.run(
                    [optimizer, loss, acc_num, acc_prob],
                    feed_dict=batch_feed_dict
                )
                print(f"\rEpoch {(epoch+1):3d}/{FLAGS.n_iter}, Batch {(batch_n):2d}: Loss = {current_loss:.4f}, Accuracy = {current_acc_prob:.1%}", end='')

        # Save the trained model
        saver = tf.train.Saver()
        saver.save(sess, FLAGS.model_path + 'model.ckpt')

    print('Training finished')

def test(test_path):
    
    tf.reset_default_graph()

    # Load your word embeddings
    word_id_mapping, w2v = load_w2v(FLAGS.test_embedding_path, FLAGS.embedding_dim)
    word_embedding = tf.constant(w2v, name='word_embedding')

    te_x, te_sen_len, te_x_bw, te_sen_len_bw, te_y, te_target_word, te_tar_len, _, _, _ = load_inputs_twitter(
        test_path, word_id_mapping, FLAGS.max_sentence_len, 'TC', True, FLAGS.max_target_len)
    
    # Create placeholders
    with tf.name_scope('inputs'):
        y = tf.placeholder(tf.float32, [None, FLAGS.n_class])

        x = tf.placeholder(tf.int32, [None, FLAGS.max_sentence_len])
        sen_len = tf.placeholder(tf.int32, None)

        x_bw = tf.placeholder(tf.int32, [None, FLAGS.max_sentence_len])
        sen_len_bw = tf.placeholder(tf.int32, [None])

        target_words = tf.placeholder(tf.int32, [None, FLAGS.max_target_len])
        tar_len = tf.placeholder(tf.int32, [None])

        kp = tf.placeholder(tf.float32)

    # Embedding lookup
    inputs_fw = tf.nn.embedding_lookup(word_embedding, x)
    inputs_bw = tf.nn.embedding_lookup(word_embedding, x_bw)
    target = tf.nn.embedding_lookup(word_embedding, target_words)

    # Define model
    prob = lcr_rot_hh(inputs_fw, inputs_bw, sen_len, sen_len_bw, target, tar_len, kp)
    
    saver = tf.train.import_meta_graph(FLAGS.model_path + 'model.ckpt.meta')

    with tf.Session() as sess:
        # Restore the saved variables
        saver.restore(sess, FLAGS.model_path + 'model.ckpt')
        # Access tensors from the graph
        graph = tf.get_default_graph()

        test_feed_dict = {
                y: te_y,
                x: te_x,
                x_bw: te_x_bw,
                sen_len: te_sen_len,
                sen_len_bw: te_sen_len_bw,
                target_words: te_target_word,
                tar_len: te_tar_len,
                kp: 1
            }

        # Compute predicted probabilities for the test set
        test_prob = sess.run(prob, feed_dict=test_feed_dict)
        correct_pred = np.equal(np.argmax(test_prob, 1), np.argmax(te_y, 1))
        test_acc_prob = np.mean(correct_pred)

        with open(FLAGS.model_path+'cp.txt', 'w') as file:
            for item in correct_pred:
                file.write(str(item) + '\n')

        print(f"Test Accuracy: {test_acc_prob:.1%}")

    print('Testing finished')


def main(_):

    train(FLAGS.train_path)
    # test(FLAGS.test_path)
    

if __name__ == '__main__':
    tf.app.run()
