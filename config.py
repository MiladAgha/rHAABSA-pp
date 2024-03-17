#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
#general variables
tf.app.flags.DEFINE_string('embedding_type','BERT','can be: glove, word2vec-cbow, word2vec-SG, fasttext, BERT, BERT_Large, ELMo')
tf.app.flags.DEFINE_integer("year",2014, "year data set [2014]")
tf.app.flags.DEFINE_integer('embedding_dim', 768, 'dimension of word embedding')
tf.app.flags.DEFINE_integer('batch_size', 20, 'number of example per batch')
tf.app.flags.DEFINE_integer('n_hidden', 300, 'number of hidden unit')
tf.app.flags.DEFINE_float('learning_rate', 0.09, 'learning rate')
tf.app.flags.DEFINE_float('momentum', 0.85, 'momentum')
tf.app.flags.DEFINE_integer('n_class', 3, 'number of distinct class')
tf.app.flags.DEFINE_integer('max_sentence_len', 150, 'max number of tokens per sentence')
tf.app.flags.DEFINE_integer('max_doc_len', 20, 'max number of tokens per sentence')
tf.app.flags.DEFINE_float('l2_reg', 0.00001, 'l2 regularization')
tf.app.flags.DEFINE_float('random_base', 0.01, 'initial random base')
tf.app.flags.DEFINE_integer('display_step', 4, 'number of test display step')
tf.app.flags.DEFINE_integer('n_iter', 200, 'number of train iter')
tf.app.flags.DEFINE_float('keep_prob', 0.3, 'dropout keep prob')
tf.app.flags.DEFINE_string('t1', 'last', 'type of hidden output')
tf.app.flags.DEFINE_string('t2', 'last', 'type of hidden output')
tf.app.flags.DEFINE_integer('n_layer', 3, 'number of stacked rnn')
tf.app.flags.DEFINE_string('is_r', '1', 'prob')
tf.app.flags.DEFINE_integer('max_target_len', 25, 'max target length')

# traindata, testdata and embeddings, train path aangepast met ELMo
tf.app.flags.DEFINE_string("train_path", 'data/programGeneratedData/'+ str(FLAGS.year) + 'train'+ str(FLAGS.embedding_type) +'.txt', "formatted train data path")
tf.app.flags.DEFINE_string("test_path", 'data/programGeneratedData/' + str(FLAGS.year) + 'test' + str(FLAGS.embedding_type) + '.txt', "formatted test data path")
tf.app.flags.DEFINE_string("embedding_path", 'data/programGeneratedData/' + str(FLAGS.year) + str(FLAGS.embedding_type) + '_emb.txt', "pre-trained glove vectors file path")
tf.app.flags.DEFINE_string("train_path_ont", "data/programGeneratedData/ont_"+ str(FLAGS.embedding_type) +str(FLAGS.embedding_dim)+"traindata"+str(FLAGS.year)+".txt", "train data path for ont")
tf.app.flags.DEFINE_string("test_path_ont", "data/programGeneratedData/ont_"+ str(FLAGS.embedding_type) +str(FLAGS.embedding_dim)+"testdata"+str(FLAGS.year)+".txt", "formatted test data path")
tf.app.flags.DEFINE_string("remaining_test_path", "data/programGeneratedData/" + str(FLAGS.embedding_type) +str(FLAGS.embedding_dim)+'remainingtestdata'+str(FLAGS.year)+".txt", "formatted remaining test data path after ontology")

#svm traindata, svm testdata
tf.app.flags.DEFINE_string("train_svm_path", "data/programGeneratedData/"+str(FLAGS.embedding_dim)+'trainsvmdata'+str(FLAGS.year)+".txt", "train data path")
tf.app.flags.DEFINE_string("test_svm_path", "data/programGeneratedData/"+str(FLAGS.embedding_dim)+'testsvmdata'+str(FLAGS.year)+".txt", "formatted test data path")
tf.app.flags.DEFINE_string("remaining_svm_test_path", "data/programGeneratedData/"+str(FLAGS.embedding_dim)+'remainingsvmtestdata'+str(FLAGS.year)+".txt", "formatted remaining test data path after ontology")

#hyper traindata, hyper testdata
tf.app.flags.DEFINE_string("hyper_train_path", "data/programGeneratedData/"+str(FLAGS.embedding_dim)+'hypertraindata'+str(FLAGS.year)+".txt", "hyper train data path")
tf.app.flags.DEFINE_string("hyper_eval_path", "data/programGeneratedData/"+str(FLAGS.embedding_dim)+'hyperevaldata'+str(FLAGS.year)+".txt", "hyper eval data path")
tf.app.flags.DEFINE_string("hyper_svm_train_path", "data/programGeneratedData/"+str(FLAGS.embedding_dim)+'hypertrainsvmdata'+str(FLAGS.year)+".txt", "hyper train svm data path")
tf.app.flags.DEFINE_string("hyper_svm_eval_path", "data/programGeneratedData/"+str(FLAGS.embedding_dim)+'hyperevalsvmdata'+str(FLAGS.year)+".txt", "hyper eval svm data path")

#external data sources
tf.app.flags.DEFINE_string("train_data", "data/externalData/restaurant_train_"+str(FLAGS.year)+".xml", "train data path")
tf.app.flags.DEFINE_string("test_data", "data/externalData/restaurant_test_"+str(FLAGS.year)+".xml", "test data path")

tf.app.flags.DEFINE_string('save_path', 'data/results/' + str(FLAGS.year) + str(FLAGS.embedding_type) , 'path to save model')
