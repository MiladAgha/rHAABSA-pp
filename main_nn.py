import numpy as np
import tensorflow as tf
import model_LCRrot

from utils import *
from config import *

def main(_):
    print("[INFO] Loading word emmeddings...")
    word_id_mapping, word_embeddings_matrix = load_w2v(FLAGS.embedding_path, FLAGS.embedding_dim)

    print("[INFO] Loading data...")
    tr_x, _, tr_x_bw, _, tr_y, tr_target_word, _, _, _, _ = load_inputs_twitter(
        FLAGS.train_path, word_id_mapping, FLAGS.max_sentence_len, 'TC', True, FLAGS.max_target_len)

    te_x, _, te_x_bw, _, te_y, te_target_word, _, _, _, _ = load_inputs_twitter(
        FLAGS.test_path, word_id_mapping, FLAGS.max_sentence_len, 'TC', True, FLAGS.max_target_len)

    trainX = [np.asarray(tr_x).astype('float32'), np.asarray(tr_target_word).astype('float32'), np.asarray(tr_x_bw).astype('float32')]
    trainY = np.asarray(tr_y).astype('float32')

    testX = [np.asarray(te_x).astype('float32'), np.asarray(te_target_word).astype('float32'), np.asarray(te_x_bw).astype('float32')]
    testY = np.asarray(te_y).astype('float32')

    print("[INFO] Building model...")
    model = model_LCRrot.create_model(word_embeddings_matrix)
    # model.summary()
    model.compile(loss="categorical_crossentropy", optimizer="adam")

    print("[INFO] Training model...")
    model.fit(x=trainX, y=trainY, 
        validation_data=(testX, testY),
        epochs=200, batch_size=20)

    print("[INFO] Saving model...")
    model.save_weights(FLAGS.save_path + '_weights')

    # print("[INFO] Loading model...")
    # model = model.load_weights(FLAGS.save_path + '_weights')

    print("[INFO] Testing model...")
    preds = model.predict(testX)

    correct_pred = np.equal(np.argmax(preds, 1), np.argmax(te_y, 1))
    with open(FLAGS.save_path + '.txt', 'w') as file:
        for item in correct_pred:
            file.write(str(item) + '\n')
            
    test_acc_prob = np.mean(correct_pred)
    print(f"Test Accuracy: {test_acc_prob:.1%}")

if __name__ == '__main__':
    tf.app.run()