#!/usr/bin/env python
# encoding: utf-8

from layer_nn import softmax_layer, bi_dynamic_rnn, reduce_mean_with_len
from layer_att import bilinear_attention_layer, dot_produce_attention_layer
from config import *

tf.set_random_seed(1)

def lcr_rot_hh(input_fw, input_bw, sen_len_fw, sen_len_bw, target, sen_len_tr, keep_prob):

    _id = 'all'
    
    cell = tf.contrib.rnn.LSTMCell
    # left hidden
    input_fw = tf.nn.dropout(tf.cast(input_fw,dtype=tf.float32), keep_prob=keep_prob)
    hiddens_l = bi_dynamic_rnn(cell, input_fw, FLAGS.n_hidden, sen_len_fw, FLAGS.max_sentence_len, 'l' + _id, 'all')
    
    # right hidden
    input_bw = tf.nn.dropout(tf.cast(input_bw,dtype=tf.float32), keep_prob=keep_prob)
    hiddens_r = bi_dynamic_rnn(cell, input_bw, FLAGS.n_hidden, sen_len_bw, FLAGS.max_sentence_len, 'r' + _id, 'all')
    
    # target hidden
    target = tf.nn.dropout(tf.cast(target,dtype=tf.float32), keep_prob=keep_prob)
    hiddens_t = bi_dynamic_rnn(cell, target, FLAGS.n_hidden, sen_len_tr, FLAGS.max_sentence_len, 't' + _id, 'all')
    pool_t = reduce_mean_with_len(hiddens_t, sen_len_tr)

    # attention left
    att_l = bilinear_attention_layer(hiddens_l, pool_t, sen_len_fw, 2 * FLAGS.n_hidden, FLAGS.l2_reg, FLAGS.random_base, 'tl')
    outputs_t_l_init = tf.matmul(att_l, hiddens_l)
    outputs_t_l = tf.squeeze(outputs_t_l_init)
    # attention right
    att_r = bilinear_attention_layer(hiddens_r, pool_t, sen_len_bw, 2 * FLAGS.n_hidden, FLAGS.l2_reg, FLAGS.random_base, 'tr')
    outputs_t_r_init = tf.matmul(att_r, hiddens_r)
    outputs_t_r = tf.squeeze(outputs_t_r_init)

    # attention target left
    att_t_l = bilinear_attention_layer(hiddens_t, outputs_t_l, sen_len_tr, 2 * FLAGS.n_hidden, FLAGS.l2_reg, FLAGS.random_base, 'l')
    outputs_l_init = tf.matmul(att_t_l, hiddens_t)
    outputs_l = tf.squeeze(outputs_l_init)
    # attention target right
    att_t_r = bilinear_attention_layer(hiddens_t, outputs_t_r, sen_len_tr, 2 * FLAGS.n_hidden, FLAGS.l2_reg, FLAGS.random_base, 'r')
    outputs_r_init = tf.matmul(att_t_r, hiddens_t)
    outputs_r = tf.squeeze(outputs_r_init)

    outputs_init_context = tf.concat([outputs_t_l_init, outputs_t_r_init], 1)
    outputs_init_target = tf.concat([outputs_l_init, outputs_r_init], 1)
    att_outputs_context = dot_produce_attention_layer(outputs_init_context, None, 2 * FLAGS.n_hidden, FLAGS.l2_reg, FLAGS.random_base, 'fin1')
    att_outputs_target = dot_produce_attention_layer(outputs_init_target, None, 2 * FLAGS.n_hidden, FLAGS.l2_reg, FLAGS.random_base, 'fin2')
    outputs_l = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_target[:,:,0], 2), outputs_l_init))
    outputs_r = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_target[:,:,1], 2), outputs_r_init))
    outputs_t_l = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_context[:,:,0], 2), outputs_t_l_init))
    outputs_t_r = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_context[:,:,1], 2), outputs_t_r_init))

    for i in range(2):
        # attention target
        att_l = bilinear_attention_layer(hiddens_l, outputs_l, sen_len_fw, 2 * FLAGS.n_hidden, FLAGS.l2_reg, FLAGS.random_base, 'tl'+str(i))
        outputs_t_l_init = tf.matmul(att_l, hiddens_l)
        outputs_t_l = tf.squeeze(outputs_t_l_init)

        att_r = bilinear_attention_layer(hiddens_r, outputs_r, sen_len_bw, 2 * FLAGS.n_hidden, FLAGS.l2_reg, FLAGS.random_base, 'tr'+str(i))
        outputs_t_r_init = tf.matmul(att_r, hiddens_r)
        outputs_t_r = tf.squeeze(outputs_t_r_init)

        # attention left
        att_t_l = bilinear_attention_layer(hiddens_t, outputs_t_l, sen_len_tr, 2 * FLAGS.n_hidden, FLAGS.l2_reg, FLAGS.random_base, 'l'+str(i))
        outputs_l_init = tf.matmul(att_t_l, hiddens_t)
        outputs_l = tf.squeeze(outputs_l_init)

        # attention right
        att_t_r = bilinear_attention_layer(hiddens_t, outputs_t_r, sen_len_tr, 2 * FLAGS.n_hidden, FLAGS.l2_reg, FLAGS.random_base, 'r'+str(i))
        outputs_r_init = tf.matmul(att_t_r, hiddens_t)
        outputs_r = tf.squeeze(outputs_r_init)

        outputs_init_context = tf.concat([outputs_t_l_init, outputs_t_r_init], 1)
        outputs_init_target = tf.concat([outputs_l_init, outputs_r_init], 1)
        att_outputs_context = dot_produce_attention_layer(outputs_init_context, None, 2 * FLAGS.n_hidden, FLAGS.l2_reg, FLAGS.random_base, 'fin1'+str(i))
        att_outputs_target = dot_produce_attention_layer(outputs_init_target, None, 2 * FLAGS.n_hidden, FLAGS.l2_reg, FLAGS.random_base, 'fin2'+str(i))
        outputs_l = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_target[:,:,0], 2), outputs_l_init))
        outputs_r = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_target[:,:,1], 2), outputs_r_init))
        outputs_t_l = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_context[:,:,0], 2), outputs_t_l_init))
        outputs_t_r = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_context[:,:,1], 2), outputs_t_r_init))

    outputs_fin = tf.concat([outputs_l, outputs_r, outputs_t_l, outputs_t_r], 1)
    prob = softmax_layer(outputs_fin, 8 * FLAGS.n_hidden, FLAGS.random_base, keep_prob, FLAGS.l2_reg, FLAGS.n_class)
    return prob