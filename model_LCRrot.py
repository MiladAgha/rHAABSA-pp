import tensorflow as tf
from keras.layers import Input, Dense, LSTM, Embedding, Dropout, Bidirectional, Flatten
from keras.models import Model
from keras.activations import softmax

from layer_att import BilinearAttentionLayer
from config import *
from utils import *

def create_model(word_embeddings_matrix):

    n_lstm = FLAGS.n_hidden
    drop_lstm = 1 - FLAGS.keep_prob

    vocab_size = word_embeddings_matrix.shape[0]
    embedding_dim = word_embeddings_matrix.shape[1]

    input_layer_l = Input(shape=(FLAGS.max_sentence_len,))
    embedding_layer_l = Embedding(input_dim=vocab_size,
                                    output_dim=embedding_dim,
                                    weights=[word_embeddings_matrix],
                                    trainable=False)(input_layer_l)
    h_l = Bidirectional(LSTM(n_lstm, return_sequences=True))(embedding_layer_l)
    h_l = Dropout(drop_lstm)(h_l)

    input_layer_c = Input(shape=(FLAGS.max_target_len,))
    embedding_layer_c = Embedding(input_dim=vocab_size,
                                    output_dim=embedding_dim,
                                    weights=[word_embeddings_matrix],
                                    trainable=False)(input_layer_c)
    h_c = Bidirectional(LSTM(n_lstm, return_sequences=True))(embedding_layer_c)
    h_c = Dropout(drop_lstm)(h_c)

    input_layer_r = Input(shape=(FLAGS.max_sentence_len,))
    embedding_layer_r = Embedding(input_dim=vocab_size,
                                    output_dim=embedding_dim,
                                    weights=[word_embeddings_matrix],
                                    trainable=False)(input_layer_r)
    h_r = Bidirectional(LSTM(n_lstm, return_sequences=True))(embedding_layer_r)
    h_r = Dropout(drop_lstm)(h_r)

    BAL_l = BilinearAttentionLayer(use_bias=True)
    BAL_r = BilinearAttentionLayer(use_bias=True)
    BAL_cl = BilinearAttentionLayer(use_bias=True)
    BAL_cr = BilinearAttentionLayer(use_bias=True)

    HAL_c = Dense(units=1, activation='tanh')
    HAL_t = Dense(units=1, activation='tanh')

    for i in range(3):
        if i == 0:
            r_cp = tf.reduce_mean(h_c, axis=1, keepdims=True)

            alfa_l = BAL_l([h_l, r_cp])
            r_l = tf.matmul(alfa_l, h_l, transpose_a=True)
            alfa_r = BAL_r([h_r, r_cp])
            r_r = tf.matmul(alfa_r, h_r, transpose_a=True)
            alfa_cl = BAL_cl([h_c, r_l])
            r_cl = tf.matmul(alfa_cl, h_c, transpose_a=True)
            alfa_cr = BAL_cr([h_c, r_r])
            r_cr = tf.matmul(alfa_cr, h_c, transpose_a=True)

            h_alfa_l, h_alfa_r = tf.split(softmax(tf.concat([HAL_c(r_l), HAL_c(r_r)], axis=1), axis=1), num_or_size_splits=2, axis=1)
            r_l = tf.matmul(h_alfa_l, r_l)
            r_r = tf.matmul(h_alfa_r, r_r)
            h_alfa_cl, h_alfa_cr = tf.split(softmax(tf.concat([HAL_t(r_cl), HAL_t(r_cr)], axis=1), axis=1), num_or_size_splits=2, axis=1)
            r_cl = tf.matmul(h_alfa_cl, r_cl)
            r_cr = tf.matmul(h_alfa_cr, r_cr)

        alfa_l = BAL_l([h_l, r_cl])
        r_l = tf.matmul(alfa_l, h_l, transpose_a=True)
        alfa_r = BAL_r([h_r, r_cr])
        r_r = tf.matmul(alfa_r, h_r, transpose_a=True)
        alfa_cl = BAL_cl([h_c, r_l])
        r_cl = tf.matmul(alfa_cl, h_c, transpose_a=True)
        alfa_cr = BAL_cr([h_c, r_r])
        r_cr = tf.matmul(alfa_cr, h_c, transpose_a=True)

        h_alfa_l, h_alfa_r = tf.split(softmax(tf.concat([HAL_c(r_l), HAL_c(r_r)], axis=1), axis=1), num_or_size_splits=2, axis=1)
        r_l = tf.matmul(h_alfa_l, r_l)
        r_r = tf.matmul(h_alfa_r, r_r)
        h_alfa_cl, h_alfa_cr = tf.split(softmax(tf.concat([HAL_t(r_cl), HAL_t(r_cr)], axis=1), axis=1), num_or_size_splits=2, axis=1)
        r_cl = tf.matmul(h_alfa_cl, r_cl)
        r_cr = tf.matmul(h_alfa_cr, r_cr)

    v = tf.concat([r_l, r_cl, r_cr, r_r], axis=2)

    p = Dense(units=3, activation='softmax')(v)
    p = Flatten()(p)

    model = Model(inputs=[input_layer_l, input_layer_c, input_layer_r], outputs=p)

    return model