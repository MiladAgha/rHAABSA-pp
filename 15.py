import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings('ignore',category=UserWarning)

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.reset_default_graph()

embedding_type = 'BERT'
embedding_dim = 768
year = 2015

keep_prob = 0.3
learning_rate = 0.09
momentum = 0.85
n_class = 3

max_sentence_len = 256
max_target_len = 32

num_epochs = 200
batch_size = 20

train_path = 'data/programGeneratedData/'+str(year)+'train'+str(embedding_type)+'.txt'
test_path = 'data/programGeneratedData/'+str(year)+'test'+'.txt'
train_embedding_path = 'data/programGeneratedData/'+str(year)+'train'+str(embedding_type)+'_emb.txt'
test_embedding_path = 'data/programGeneratedData/'+str(year)+'test'+'_emb.txt'
results_path = 'data/results/'+str(year)+'.txt'

def load_w2v(w2v_file, embedding_dim, is_skip=False):
    fp = open(w2v_file)
    if is_skip:
        fp.readline()
    w2v = []
    word_dict = dict()
    # [0,0,...,0] represent absent words
    w2v.append([0.] * embedding_dim)
    cnt = 0
    for line in fp:
        cnt += 1
        line = line.split()
        # line = line.split()
        if len(line) != embedding_dim + 1:
            print('a bad word embedding: {}'.format(line[0]))
            continue
        w2v.append([float(v) for v in line[1:]])
        word_dict[line[0]] = cnt
    w2v = np.asarray(w2v, dtype=np.float32)
    w2v = np.row_stack((w2v, np.sum(w2v, axis=0) / cnt))
    # print(np.shape(w2v))
    word_dict['$t$'] = (cnt + 1)
    # w2v -= np.mean(w2v, axis=0)
    # w2v /= np.std(w2v, axis=0)
    # print(word_dict['$t$'], len(w2v))
    return word_dict, w2v

def load_word_id_mapping(word_id_file, encoding='utf8'):
    """
    :param word_id_file: word-id mapping file path
    :param encoding: file's encoding, for changing to unicode
    :return: word-id mapping, like hello=5
    """
    word_to_id = dict()
    for line in open(word_id_file):
        line = line.decode(encoding, 'ignore').lower().split()
        word_to_id[line[0]] = int(line[1])
    # print('\nload word-id mapping done!\n')
    return word_to_id

def change_y_to_onehot(y):
    from collections import Counter
    # print(Counter(y))
    class_set = set(y)
    n_class = len(class_set)
    y_onehot_mapping = dict(zip(class_set, range(n_class)))
    # print(y_onehot_mapping)
    onehot = []
    for label in y:
        tmp = [0] * n_class
        tmp[y_onehot_mapping[label]] = 1
        onehot.append(tmp)
    return np.asarray(onehot, dtype=np.int32)

def load_inputs_twitter(input_file, word_id_file, sentence_len, type_='', is_r=True, target_len=10, encoding='utf8'):
    if type(word_id_file) is str:
        word_to_id = load_word_id_mapping(word_id_file)
    else:
        word_to_id = word_id_file
    # print('load word-to-id done!')

    x, y, sen_len = [], [], []
    x_r, sen_len_r = [], []
    target_words = []
    tar_len = []
    all_target, all_sent, all_y = [], [], []
    # read in txt file
    lines = open(input_file).readlines()
    for i in range(0, len(lines), 3):
        # targets
        words = lines[i + 1].lower().split()
        target = words

        target_word = []
        for w in words:
            if w in word_to_id:
                target_word.append(word_to_id[w])
        l = min(len(target_word), target_len)
        tar_len.append(l)
        target_words.append(target_word[:l] + [0] * (target_len - l))

        # sentiment
        y.append(lines[i + 2].strip().split()[0])

        # left and right context
        words = lines[i].lower().split()
        sent = words
        words_l, words_r = [], []
        flag = True
        for word in words:
            if word == '$t$':
                flag = False
                continue
            if flag:
                if word in word_to_id:
                    words_l.append(word_to_id[word])
            else:
                if word in word_to_id:
                    words_r.append(word_to_id[word])
        if type_ == 'TD' or type_ == 'TC':
            # words_l.extend(target_word)
            words_l = words_l[:sentence_len]
            words_r = words_r[:sentence_len]
            sen_len.append(len(words_l))
            x.append(words_l + [0] * (sentence_len - len(words_l)))
            # tmp = target_word + words_r
            tmp = words_r
            if is_r:
                tmp.reverse()
            sen_len_r.append(len(tmp))
            x_r.append(tmp + [0] * (sentence_len - len(tmp)))
            all_sent.append(sent)
            all_target.append(target)
        else:
            words = words_l + target_word + words_r
            words = words[:sentence_len]
            sen_len.append(len(words))
            x.append(words + [0] * (sentence_len - len(words)))
    all_y = y;
    y = change_y_to_onehot(y)
    if type_ == 'TD':
        return np.asarray(x, dtype="object"), np.asarray(sen_len, dtype="object"), np.asarray(x_r, dtype="object"), \
               np.asarray(sen_len_r, dtype="object"), np.asarray(y, dtype="object")
    elif type_ == 'TC':
        return np.asarray(x, dtype="object"), np.asarray(sen_len, dtype="object"), np.asarray(x_r, dtype="object"), np.asarray(sen_len_r, dtype="object"), \
               np.asarray(y, dtype="object"), np.asarray(target_words, dtype="object"), np.asarray(tar_len, dtype="object"), np.asarray(all_sent, dtype="object"), np.asarray(all_target, dtype="object"), np.asarray(all_y, dtype="object")
    elif type_ == 'IAN':
        return np.asarray(x, dtype="object"), np.asarray(sen_len, dtype="object"), np.asarray(target_words, dtype="object"), \
               np.asarray(tar_len, dtype="object"), np.asarray(y, dtype="object")
    else:
        return np.asarray(x, dtype="object"), np.asarray(sen_len, dtype="object"), np.asarray(y, dtype="object")

def batch_index(length, batch_size, n_iter=100, is_shuffle=True):
    index = list(range(length))
    for j in range(n_iter):
        if is_shuffle:
            np.random.shuffle(index)
        for i in range(int(length / batch_size) + (1 if length % batch_size else 0)):
            yield index[i * batch_size:(i + 1) * batch_size]

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

#------------------------------------------------------------------------------

def reduce_mean_with_len(inputs, length):
    """
    :param inputs: 3-D tensor
    :param length: the length of dim [1]
    :return: 2-D tensor
    """
    length = tf.cast(tf.reshape(length, [-1, 1]), tf.float32) + 1e-9
    inputs = tf.reduce_sum(inputs, 1, keepdims=False) / length
    return inputs

def bi_dynamic_rnn(cell, inputs, n_hidden, length, max_len, scope_name, out_type='last'):
    outputs, state = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=cell(n_hidden),
        cell_bw=cell(n_hidden),
        inputs=inputs,
        sequence_length=length,
        dtype=tf.float32,
        scope=scope_name
    )
    if out_type == 'last':
        outputs_fw, outputs_bw = outputs
        outputs_bw = tf.reverse_sequence(outputs_bw, tf.cast(length, tf.int64), seq_dim=1)
        outputs = tf.concat([outputs_fw, outputs_bw], 2)
    else:
        outputs = tf.concat(outputs, 2)  # batch_size * max_len * 2n_hidden
    batch_size = tf.shape(outputs)[0]
    if out_type == 'last':
        index = tf.range(0, batch_size) * max_len + (length - 1)
        outputs = tf.gather(tf.reshape(outputs, [-1, 2 * n_hidden]), index)  # batch_size * 2n_hidden
    elif out_type == 'all_avg':
        outputs = reduce_mean_with_len(outputs, length)  # batch_size * 2n_hidden
    return outputs

def softmax_layer(inputs, n_hidden, random_base, keep_prob, l2_reg, n_class, scope_name='1'):
    w = tf.get_variable(
        name='softmax_w' + scope_name,
        shape=[n_hidden, n_class],
        initializer=tf.random_normal_initializer(mean=0., stddev=np.sqrt(2. / (n_hidden + n_class))),
        # initializer=tf.random_uniform_initializer(-random_base, random_base),
        # initializer=tf.random_uniform_initializer(-np.sqrt(6.0 / (n_hidden + n_class)), np.sqrt(6.0 / (n_hidden + n_class))),
        regularizer=tf.keras.regularizers.L2(l2_reg)
    )
    b = tf.get_variable(
        name='softmax_b' + scope_name,
        shape=[n_class],
        initializer=tf.random_normal_initializer(mean=0., stddev=np.sqrt(2. / (n_class))),
        # initializer=tf.random_uniform_initializer(-random_base, random_base),
        regularizer=tf.keras.regularizers.L2(l2_reg)
    )
    with tf.name_scope('softmax'):
        outputs = tf.nn.dropout(inputs, keep_prob=keep_prob)
        predict = tf.matmul(outputs, w) + b
        predict = tf.nn.softmax(predict)
    return predict

#-------------------------------------------------------------------------------

def bilinear_attention_layer(inputs, attend, length, n_hidden, l2_reg, random_base, layer_id=1):
    """
    :param inputs: batch * max_len * n_hidden
    :param attend: batch * n_hidden
    :param length:
    :param n_hidden:
    :param l2_reg:
    :param random_base:
    :param layer_id:
    :return:
    """
    batch_size = tf.shape(inputs)[0]
    max_len = tf.shape(inputs)[1]
    shapes = inputs.shape.as_list()
    w = tf.get_variable(
        name='att_w_' + str(layer_id),
        shape=[n_hidden, n_hidden],
        # initializer=tf.random_normal_initializer(mean=0., stddev=np.sqrt(2. / (n_hidden + n_hidden))),
        initializer=tf.random_uniform_initializer(-random_base, random_base),
        # initializer=tf.random_uniform_initializer(-np.sqrt(6.0 / (n_hidden + n_hidden)), np.sqrt(6.0 / (n_hidden + n_hidden))),
        regularizer=tf.keras.regularizers.L2(l2_reg)
    )
    b = tf.get_variable(
        name='att_b' + str(layer_id),
        shape=[shapes[1]],
        # initializer=tf.random_normal_initializer(mean=0.0, stddev=np.sqrt(2. / (n_hidden + n_hidden))),
        initializer=tf.random_uniform_initializer(-0., 0.),
        regularizer=tf.keras.regularizers.L2(l2_reg)
    )
    inputs = tf.reshape(inputs, [-1, n_hidden])
    tmp = tf.reshape(tf.matmul(inputs, w), [-1, max_len, n_hidden])
    attend = tf.expand_dims(attend, 2)
    #tmp = tf.reshape(tf.tanh(tf.squeeze(tf.matmul(tmp, attend))+b), [batch_size, 1, max_len]) 
    tmp = tf.reshape(tf.matmul(tmp, attend), [batch_size, 1, max_len])
    #tmp = tf.tanh(tmp + b)
    # M = tf.expand_dims(tf.matmul(attend, w), 2)
    # tmp = tf.reshape(tf.batch_matmul(inputs, M), [batch_size, 1, max_len])
    return softmax_with_len(tmp, length, max_len)

def dot_produce_attention_layer(inputs, length, n_hidden, l2_reg, random_base, layer_id=1):
    """
    :param inputs: batch * max_len * n_hidden
    :param length: batch * 1
    :param n_hidden:
    :param l2_reg:
    :param random_base:
    :param layer_id: layer's identical id
    :return: batch * 1 * max_len
    """
    batch_size = tf.shape(inputs)[0]
    max_len = tf.shape(inputs)[1]
    shapes = inputs.shape.as_list()
    u = tf.get_variable(
        name='att_u_' + str(layer_id),
        shape=[n_hidden, 1],
        initializer=tf.random_normal_initializer(mean=0., stddev=np.sqrt(2. / (n_hidden + 1))),
        #initializer=tf.random_uniform_initializer(-random_base, random_base),
        # initializer=tf.random_uniform_initializer(-np.sqrt(6.0 / (n_hidden + 1)), np.sqrt(6.0 / (n_hidden + 1))),
        regularizer=tf.keras.regularizers.L2(l2_reg)
    )
    b = tf.get_variable(
        name='att_b' + str(layer_id),
        shape=[shapes[1]],
        # initializer=tf.random_normal_initializer(mean=0.0, stddev=np.sqrt(2. / (n_hidden + n_hidden))),
        initializer=tf.random_uniform_initializer(-0., 0.),
        regularizer=tf.keras.regularizers.L2(l2_reg)
    )
    inputs = tf.reshape(inputs, [-1, n_hidden])
    tmp = tf.reshape(tf.matmul(inputs, u), [batch_size, 1, max_len])
    #tmp = tf.tanh(tmp + b)
    alpha = softmax_with_len(tmp, length, max_len)
    return alpha

def softmax_with_len(inputs, length, max_len):
    inputs = tf.cast(inputs, tf.float32)
    # max_axis = tf.reduce_max(inputs, -1, keep_dims=True)
    # inputs = tf.exp(inputs - max_axis)
    inputs = tf.exp(inputs)
    if length is not None:
        length = tf.reshape(length, [-1])
        mask = tf.reshape(tf.cast(tf.sequence_mask(length, max_len), tf.float32), tf.shape(inputs))
        inputs *= mask
    _sum = tf.reduce_sum(inputs, reduction_indices=-1, keepdims=True) + 1e-9
    return inputs / _sum

#------------------------------------------------------------------------------

def lcr_rot(input_fw, input_bw, sen_len_fw, sen_len_bw, target, sen_len_tr, keep_prob):

    max_sentence_len = 256
    n_hidden = 300
    n_class = 3
    random_base = 0.01
    l2  =  0.00001
    _id = 'all'
    
    cell = tf.nn.rnn_cell.LSTMCell
    # left hidden
    input_fw = tf.nn.dropout(tf.cast(input_fw,dtype=tf.float32), keep_prob=keep_prob)
    hiddens_l = bi_dynamic_rnn(cell, input_fw, n_hidden, sen_len_fw, max_sentence_len, 'l' + _id, 'all')
    
    # right hidden
    input_bw = tf.nn.dropout(tf.cast(input_bw,dtype=tf.float32), keep_prob=keep_prob)
    hiddens_r = bi_dynamic_rnn(cell, input_bw, n_hidden, sen_len_bw, max_sentence_len, 'r' + _id, 'all')
    
    # target hidden
    target = tf.nn.dropout(tf.cast(target,dtype=tf.float32), keep_prob=keep_prob)
    hiddens_t = bi_dynamic_rnn(cell, target, n_hidden, sen_len_tr, max_sentence_len, 't' + _id, 'all')
    pool_t = reduce_mean_with_len(hiddens_t, sen_len_tr)

    # attention left
    att_l = bilinear_attention_layer(hiddens_l, pool_t, sen_len_fw, 2 * n_hidden, l2, random_base, 'tl')
    outputs_t_l_init = tf.matmul(att_l, hiddens_l)
    outputs_t_l = tf.squeeze(outputs_t_l_init)
    # attention right
    att_r = bilinear_attention_layer(hiddens_r, pool_t, sen_len_bw, 2 * n_hidden, l2, random_base, 'tr')
    outputs_t_r_init = tf.matmul(att_r, hiddens_r)
    outputs_t_r = tf.squeeze(outputs_t_r_init)

    # attention target left
    att_t_l = bilinear_attention_layer(hiddens_t, outputs_t_l, sen_len_tr, 2 * n_hidden, l2, random_base, 'l')
    outputs_l_init = tf.matmul(att_t_l, hiddens_t)
    outputs_l = tf.squeeze(outputs_l_init)
    # attention target right
    att_t_r = bilinear_attention_layer(hiddens_t, outputs_t_r, sen_len_tr, 2 * n_hidden, l2, random_base, 'r')
    outputs_r_init = tf.matmul(att_t_r, hiddens_t)
    outputs_r = tf.squeeze(outputs_r_init)

    outputs_init_context = tf.concat([outputs_t_l_init, outputs_t_r_init], 1)
    outputs_init_target = tf.concat([outputs_l_init, outputs_r_init], 1)
    att_outputs_context = dot_produce_attention_layer(outputs_init_context, None, 2 * n_hidden, l2, random_base, 'fin1')
    att_outputs_target = dot_produce_attention_layer(outputs_init_target, None, 2 * n_hidden, l2, random_base, 'fin2')
    outputs_l = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_target[:,:,0], 2), outputs_l_init))
    outputs_r = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_target[:,:,1], 2), outputs_r_init))
    outputs_t_l = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_context[:,:,0], 2), outputs_t_l_init))
    outputs_t_r = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_context[:,:,1], 2), outputs_t_r_init))

    for i in range(2):
        # attention target
        att_l = bilinear_attention_layer(hiddens_l, outputs_l, sen_len_fw, 2 * n_hidden, l2, random_base, 'tl'+str(i))
        outputs_t_l_init = tf.matmul(att_l, hiddens_l)
        outputs_t_l = tf.squeeze(outputs_t_l_init)

        att_r = bilinear_attention_layer(hiddens_r, outputs_r, sen_len_bw, 2 * n_hidden, l2, random_base, 'tr'+str(i))
        outputs_t_r_init = tf.matmul(att_r, hiddens_r)
        outputs_t_r = tf.squeeze(outputs_t_r_init)

        # attention left
        att_t_l = bilinear_attention_layer(hiddens_t, outputs_t_l, sen_len_tr, 2 * n_hidden, l2, random_base, 'l'+str(i))
        outputs_l_init = tf.matmul(att_t_l, hiddens_t)
        outputs_l = tf.squeeze(outputs_l_init)

        # attention right
        att_t_r = bilinear_attention_layer(hiddens_t, outputs_t_r, sen_len_tr, 2 * n_hidden, l2, random_base, 'r'+str(i))
        outputs_r_init = tf.matmul(att_t_r, hiddens_t)
        outputs_r = tf.squeeze(outputs_r_init)

        outputs_init_context = tf.concat([outputs_t_l_init, outputs_t_r_init], 1)
        outputs_init_target = tf.concat([outputs_l_init, outputs_r_init], 1)
        att_outputs_context = dot_produce_attention_layer(outputs_init_context, None, 2 * n_hidden, l2, random_base, 'fin1'+str(i))
        att_outputs_target = dot_produce_attention_layer(outputs_init_target, None, 2 * n_hidden, l2, random_base, 'fin2'+str(i))
        outputs_l = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_target[:,:,0], 2), outputs_l_init))
        outputs_r = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_target[:,:,1], 2), outputs_r_init))
        outputs_t_l = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_context[:,:,0], 2), outputs_t_l_init))
        outputs_t_r = tf.squeeze(tf.matmul(tf.expand_dims(att_outputs_context[:,:,1], 2), outputs_t_r_init))

    outputs_fin = tf.concat([outputs_l, outputs_r, outputs_t_l, outputs_t_r], 1)
    prob = softmax_layer(outputs_fin, 8 * n_hidden, random_base, keep_prob, l2, n_class)
    return prob

def loss_func(y, prob):
    reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = - tf.reduce_mean(y * tf.log(prob)) + sum(reg_loss)
    return loss

def acc_func(y, prob):
    correct_pred = tf.equal(tf.argmax(prob, 1), tf.argmax(y, 1))
    acc_num = tf.reduce_sum(tf.cast(correct_pred, tf.int32))
    acc_prob = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return acc_num, acc_prob

#-------------------------------------------------------------------------------

tf.reset_default_graph()

# Load your word embeddings
word_id_mapping, w2v = load_w2v(train_embedding_path, embedding_dim)
word_embedding = tf.constant(w2v, name='word_embedding')

tr_x, tr_sen_len, tr_x_bw, tr_sen_len_bw, tr_y, tr_target_word, tr_tar_len, _, _, _ = load_inputs_twitter(
    train_path, word_id_mapping, max_sentence_len, 'TC', True, max_target_len)

# Create placeholders
y = tf.placeholder(tf.float32, [None, n_class])

x = tf.placeholder(tf.int32, [None, max_sentence_len])
sen_len = tf.placeholder(tf.int32, None)

x_bw = tf.placeholder(tf.int32, [None, max_sentence_len])
sen_len_bw = tf.placeholder(tf.int32, [None])

target_words = tf.placeholder(tf.int32, [None, max_target_len])
tar_len = tf.placeholder(tf.int32, [None])

kp = tf.placeholder(tf.float32)

# Embedding lookup
inputs_fw = tf.nn.embedding_lookup(word_embedding, x)
inputs_bw = tf.nn.embedding_lookup(word_embedding, x_bw)
target = tf.nn.embedding_lookup(word_embedding, target_words)

# Define model
prob = lcr_rot(inputs_fw, inputs_bw, sen_len, sen_len_bw, target, tar_len, kp)

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

    for epoch in range(num_epochs):
        batch_n = 0
        for batch_feed_dict, batch_len in get_batch_data(tr_x, tr_sen_len, tr_x_bw, tr_sen_len_bw, tr_y, tr_target_word, tr_tar_len, batch_size, keep_prob):
            batch_n += 1
            # Run a training step
            _, current_loss, current_acc_num, current_acc_prob = sess.run(
                [optimizer, loss, acc_num, acc_prob],
                feed_dict=batch_feed_dict
            )
            print(f"\rEpoch {(epoch+1):3d}/{num_epochs}, Batch {(batch_n):2d}: Loss = {current_loss:.4f}, Accuracy = {current_acc_prob:.1%}", end='')

    print('\nTraining finished')

    word_id_mapping, w2v = load_w2v(test_embedding_path, embedding_dim)
    word_embedding = tf.constant(w2v, name='word_embedding')

    tr_x, tr_sen_len, tr_x_bw, tr_sen_len_bw, tr_y, tr_target_word, tr_tar_len, _, _, _ = load_inputs_twitter(
        train_path, word_id_mapping, max_sentence_len, 'TC', True, max_target_len)
    te_x, te_sen_len, te_x_bw, te_sen_len_bw, te_y, te_target_word, te_tar_len, _, _, _ = load_inputs_twitter(
        test_path, word_id_mapping, max_sentence_len, 'TC', True, max_target_len)

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

    test_prob = sess.run(prob, feed_dict=test_feed_dict)
    correct_pred = np.equal(np.argmax(test_prob, 1), np.argmax(te_y, 1))
    test_acc_prob = np.mean(correct_pred)

    with open(results_path, 'w') as file:
        for item in correct_pred:
            file.write(str(item) + '\n')

    print(f"Test Accuracy: {test_acc_prob:.1%}")