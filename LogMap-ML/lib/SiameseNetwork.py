import os
import sys
import datetime
import tensorflow as tf
import numpy as np
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn

# for Tensorflow 1.x
# from tensorflow.contrib.rnn import GRUCell

# for Tensorflow 2.x
from tensorflow.keras.layers import GRUCell


# from tensorflow.nn import static_bidirectional_rnn as bi_rnn


# Train a neural network
def siamese_nn_train(train_x1, train_x2, y_train, PARAMETERS):
    y_train = y_train[:, 1]
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            if PARAMETERS.nn_type == 'SiameseBiRNN':
                nn = SiameseBiRNN(sequence_length1=train_x1.shape[1], sequence_length2=train_x2.shape[1],
                                  channel_num=train_x1.shape[2], rnn_hidden_size=PARAMETERS.rnn_hidden_size)
            elif PARAMETERS.nn_type == 'SiameseAttBiRNN':
                nn = SiameseAttBiRNN(sequence_length1=train_x1.shape[1], sequence_length2=train_x2.shape[1],
                                     channel_num=train_x1.shape[2], rnn_hidden_size=PARAMETERS.rnn_hidden_size,
                                     attention_size=PARAMETERS.rnn_attention_size)
            elif PARAMETERS.nn_type == 'SiameseMLP':
                nn = SiameseMLP(sequence_length1=train_x1.shape[1], sequence_length2=train_x2.shape[1],
                                channel_num=train_x1.shape[2], hidden_size=PARAMETERS.mlp_hidden_size)
            else:
                print('unknown %s' % PARAMETERS.nn_type)
                sys.exit(0)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(nn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step, name="train_op")

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", nn.loss)
            acc_summary = tf.summary.scalar("accuracy", nn.accuracy)
            train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            train_summary_dir = os.path.join(PARAMETERS.nn_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Checkpoint directory
            checkpoint_dir = os.path.abspath(os.path.join(PARAMETERS.nn_dir, "checkpoints"))
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(train_x1_batch, train_x2_batch, train_y_batch):
                feed_dict = {
                    nn.input_x1: train_x1_batch,
                    nn.input_x2: train_x2_batch,
                    nn.input_y: train_y_batch,
                    nn.dropout_keep_prob: 0.5
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, nn.loss, nn.accuracy], feed_dict)
                time_str = datetime.datetime.now().isoformat()
                if step % PARAMETERS.evaluate_every == 0:
                    print("\t {}: step {}, train loss {:g}, train acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def train_evaluate(train_x1_all, train_x2_all, train_y_all):
                feed_dict = {
                    nn.input_x1: train_x1_all,
                    nn.input_x2: train_x2_all,
                    nn.input_y: train_y_all,
                    nn.dropout_keep_prob: 0.5
                }
                loss, accuracy = sess.run([nn.loss, nn.accuracy], feed_dict)
                print("\t train loss {:g}, train acc {:g}".format(loss, accuracy))

            batches = batch_iter(list(zip(train_x1, train_x2, y_train)), PARAMETERS.num_epochs, PARAMETERS.batch_size)
            current_step = 0
            for batch in batches:
                x1_batch, x2_batch, y_batch = zip(*batch)
                train_step(train_x1_batch=x1_batch, train_x2_batch=x2_batch, train_y_batch=y_batch)
                current_step = tf.train.global_step(sess, global_step)

            train_evaluate(train_x1_all=train_x1, train_x2_all=train_x2, train_y_all=y_train)
            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            print("\t Saved model checkpoint to {}\n".format(os.path.basename(path)))


# Predict with a trained neural network
# input: test_x1, test_x2
# output: test_p (an array, each item represents the score of one sample)
def siamese_nn_predict(test_x1, test_x2, nn_dir):
    checkpoint_dir = os.path.join(nn_dir, 'checkpoints')
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)

        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)
            input_x1 = graph.get_operation_by_name("input_x1").outputs[0]
            input_x2 = graph.get_operation_by_name("input_x2").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            distance = graph.get_operation_by_name("output/distance").outputs[0]

            # predict
            test_distance = sess.run(distance, {input_x1: test_x1, input_x2: test_x2, dropout_keep_prob: 1.0})

    return test_distance


# Generate batches of the samples
# In each epoch, samples are traversed one time batch by batch
def batch_iter(data, num_epochs, batch_size, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches = int((data_size - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        if shuffle:
            batch_shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[batch_shuffle_indices]
        else:
            shuffled_data = data

        if num_batches > 0:
            for batch_num in range(num_batches):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]
        else:
            yield shuffled_data


class SiameseMLP(object):
    def __init__(self, sequence_length1, sequence_length2, channel_num, hidden_size):
        # Placeholders for input, output and dropout
        self.input_x1 = tf.placeholder(tf.float32, [None, sequence_length1, channel_num], name="input_x1")
        self.input_x2 = tf.placeholder(tf.float32, [None, sequence_length2, channel_num], name="input_x2")
        self.input_y = tf.placeholder(tf.float32, [None], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        with tf.name_scope("FCLayers"):
            FC_W11 = tf.get_variable("FC_W11", shape=[sequence_length1 * channel_num, hidden_size],
                                     initializer=tf.contrib.layers.xavier_initializer())
            FC_b11 = tf.Variable(tf.constant(0.1, shape=[hidden_size]), name="FC_b11")
            self.fc_out11 = tf.nn.xw_plus_b(tf.reshape(self.input_x1, [-1, sequence_length1 * channel_num]),
                                            FC_W11, FC_b11, name="FC_out11")
            '''FC_W12 = tf.get_variable("FC_W12", shape=[hidden_size, hidden_size],
                                     initializer=tf.contrib.layers.xavier_initializer())
            FC_b12 = tf.Variable(tf.constant(0.1, shape=[hidden_size]), name="FC_b12")
            self.fc_out12 = tf.nn.xw_plus_b(self.fc_out11, FC_W12, FC_b12, name="FC_out12")
            '''

            FC_W21 = tf.get_variable("FC_W21", shape=[sequence_length2 * channel_num, hidden_size],
                                     initializer=tf.contrib.layers.xavier_initializer())
            FC_b21 = tf.Variable(tf.constant(0.1, shape=[hidden_size]), name="FC_b21")
            self.fc_out21 = tf.nn.xw_plus_b(tf.reshape(self.input_x2, [-1, sequence_length2 * channel_num]),
                                            FC_W21, FC_b21, name="FC_out21")
            '''FC_W22 = tf.get_variable("FC_W22", shape=[hidden_size, hidden_size],
                                     initializer=tf.contrib.layers.xavier_initializer())
            FC_b22 = tf.Variable(tf.constant(0.1, shape=[hidden_size]), name="FC_b22")
            self.fc_out22 = tf.nn.xw_plus_b(self.fc_out21, FC_W22, FC_b22, name="FC_out22")
            '''

        with tf.name_scope("output"):
            self.distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.fc_out11, self.fc_out21)), 1, keep_dims=True))
            self.denominator = tf.add(tf.sqrt(tf.reduce_sum(tf.square(self.fc_out11), 1, keep_dims=True)),
                                      tf.sqrt(tf.reduce_sum(tf.square(self.fc_out21), 1, keep_dims=True)))
            self.distance = tf.div(self.distance, self.denominator)
            self.distance = tf.reshape(self.distance, [-1], name="distance")

        # Contrastive loss
        with tf.name_scope("loss"):
            item1 = self.input_y * tf.square(self.distance)
            item2 = (1 - self.input_y) * tf.square(tf.maximum((1 - self.distance), 0))
            self.loss = tf.reduce_sum(item1 + item2) / 2

        with tf.name_scope("accuracy"):
            self.temp_sim = tf.subtract(tf.ones_like(self.distance), tf.rint(self.distance), name="temp_sim")
            correct_predictions = tf.equal(self.temp_sim, self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


class SiameseBiRNN(object):
    def __init__(self, sequence_length1, sequence_length2, channel_num, rnn_hidden_size):
        # Placeholders for input, output and dropout
        self.input_x1 = tf.placeholder(tf.float32, [None, sequence_length1, channel_num], name="input_x1")
        self.input_x2 = tf.placeholder(tf.float32, [None, sequence_length2, channel_num], name="input_x2")
        self.input_y = tf.placeholder(tf.float32, [None], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        with tf.name_scope("NBiRNN"), tf.variable_scope("VBiRNN", reuse=tf.AUTO_REUSE):
            self.rnn_outputs1, _ = bi_rnn(GRUCell(rnn_hidden_size), GRUCell(rnn_hidden_size),
                                          self.input_x1, dtype=tf.float32)
            self.rnn_output1 = tf.concat(self.rnn_outputs1, 2)
            self.rnn_output_mean1 = tf.reduce_mean(self.rnn_output1, axis=1)

            self.rnn_outputs2, _ = bi_rnn(GRUCell(rnn_hidden_size), GRUCell(rnn_hidden_size),
                                          inputs=self.input_x2, dtype=tf.float32)
            self.rnn_output2 = tf.concat(self.rnn_outputs2, 2)
            self.rnn_output_mean2 = tf.reduce_mean(self.rnn_output2, axis=1)

        with tf.name_scope("output"):
            self.distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.rnn_output_mean1, self.rnn_output_mean2)),
                                                  1, keep_dims=True))
            self.denominator = tf.add(tf.sqrt(tf.reduce_sum(tf.square(self.rnn_output_mean1), 1, keep_dims=True)),
                                      tf.sqrt(tf.reduce_sum(tf.square(self.rnn_output_mean2), 1, keep_dims=True)))
            self.distance = tf.div(self.distance, self.denominator)
            self.distance = tf.reshape(self.distance, [-1], name="distance")

        # Contrastive loss
        with tf.name_scope("loss"):
            item1 = self.input_y * tf.square(self.distance)
            item2 = (1 - self.input_y) * tf.square(tf.maximum((1 - self.distance), 0))
            self.loss = tf.reduce_sum(item1 + item2) / 2

        with tf.name_scope("accuracy"):
            self.temp_sim = tf.subtract(tf.ones_like(self.distance), tf.rint(self.distance), name="temp_sim")
            correct_predictions = tf.equal(self.temp_sim, self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


class SiameseAttBiRNN(object):
    @staticmethod
    def attention(inputs, attention_size, time_major=False, return_alphas=True):
        if isinstance(inputs, tuple):
            inputs = tf.concat(inputs, 2)
        if time_major:
            inputs = tf.array_ops.transpose(inputs, [1, 0, 2])
        hidden_size = inputs.shape[2].value
        w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1), name="w_omega")
        b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1), name="b_omega")
        u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1), name="u_omega")
        with tf.name_scope('v'):
            v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)
        vu = tf.tensordot(v, u_omega, axes=1, name='vu')
        alphas = tf.nn.softmax(vu, name='alphas')
        # output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)
        output = inputs * tf.expand_dims(alphas, -1)
        output = tf.reshape(output, [-1, output.shape[1] * output.shape[2]])
        return output if not return_alphas else output, alphas

    def __init__(self, sequence_length1, sequence_length2, channel_num, rnn_hidden_size, attention_size):
        # Placeholders for input, output and dropout
        self.input_x1 = tf.placeholder(tf.float32, [None, sequence_length1, channel_num], name="input_x1")
        self.input_x2 = tf.placeholder(tf.float32, [None, sequence_length2, channel_num], name="input_x2")
        self.input_y = tf.placeholder(tf.float32, [None], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        with tf.name_scope("NBiRNN"), tf.variable_scope("VBiRNN", reuse=tf.AUTO_REUSE):
            self.rnn_outputs1, _ = bi_rnn(GRUCell(rnn_hidden_size), GRUCell(rnn_hidden_size),
                                          self.input_x1, dtype=tf.float32)
            self.att_output1, alphas1 = self.attention(inputs=self.rnn_outputs1, attention_size=attention_size)
            self.att_drop1 = tf.nn.dropout(self.att_output1, self.dropout_keep_prob, name="dropout1")
            FC_W1 = tf.get_variable("FC_W1", shape=[sequence_length1 * rnn_hidden_size * 2, 100],
                                    initializer=tf.contrib.layers.xavier_initializer())
            FC_b1 = tf.Variable(tf.constant(0.1, shape=[100]), name="FC_b1")
            self.fc_out1 = tf.nn.xw_plus_b(self.att_drop1, FC_W1, FC_b1, name="FC_out1")

            self.rnn_outputs2, _ = bi_rnn(GRUCell(rnn_hidden_size), GRUCell(rnn_hidden_size),
                                          self.input_x2, dtype=tf.float32)
            self.att_output2, alphas2 = self.attention(inputs=self.rnn_outputs2, attention_size=attention_size)
            self.att_drop2 = tf.nn.dropout(self.att_output2, self.dropout_keep_prob, name="dropout2")
            FC_W2 = tf.get_variable("FC_W2", shape=[sequence_length2 * rnn_hidden_size * 2, 100],
                                    initializer=tf.contrib.layers.xavier_initializer())
            FC_b2 = tf.Variable(tf.constant(0.1, shape=[100]), name="FC_b2")
            self.fc_out2 = tf.nn.xw_plus_b(self.att_drop2, FC_W2, FC_b2, name="FC_out2")

        with tf.name_scope("output"):
            self.distance = tf.sqrt(
                tf.reduce_sum(tf.square(tf.subtract(self.fc_out1, self.fc_out2)), 1, keep_dims=True))
            self.denominator = tf.add(tf.sqrt(tf.reduce_sum(tf.square(self.fc_out1), 1, keep_dims=True)),
                                      tf.sqrt(tf.reduce_sum(tf.square(self.fc_out2), 1, keep_dims=True)))
            self.distance = tf.div(self.distance, self.denominator)
            self.distance = tf.reshape(self.distance, [-1], name="distance")

        # Contrastive loss
        with tf.name_scope("loss"):
            item1 = self.input_y * tf.square(self.distance)
            item2 = (1 - self.input_y) * tf.square(tf.maximum((1 - self.distance), 0))
            self.loss = tf.reduce_sum(item1 + item2) / 2

        with tf.name_scope("accuracy"):
            self.temp_sim = tf.subtract(tf.ones_like(self.distance), tf.rint(self.distance), name="temp_sim")
            correct_predictions = tf.equal(self.temp_sim, self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


class SiameseLSTM(object):
    def BiRNN(self, x, dropout, scope, hidden_units):
        n_hidden = hidden_units
        n_layers = 1
        x = tf.unstack(tf.transpose(x, perm=[1, 0, 2]))
        print(x)
        with tf.name_scope("fw" + scope), tf.variable_scope("fw" + scope):
            stacked_rnn_fw = []
            for _ in range(n_layers):
                fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
                lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=dropout)
                stacked_rnn_fw.append(lstm_fw_cell)
            lstm_fw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_fw, state_is_tuple=True)

        with tf.name_scope("bw" + scope), tf.variable_scope("bw" + scope):
            stacked_rnn_bw = []
            for _ in range(n_layers):
                bw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
                lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=dropout)
                stacked_rnn_bw.append(lstm_bw_cell)
            lstm_bw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_bw, state_is_tuple=True)

        with tf.name_scope("bw" + scope), tf.variable_scope("bw" + scope):
            outputs, _, _ = tf.nn.static_bidirectional_rnn(lstm_fw_cell_m, lstm_bw_cell_m, x, dtype=tf.float32)
        return outputs[-1]

    def contrastive_loss(self, y, d):
        tmp = y * tf.square(d)
        tmp2 = (1 - y) * tf.square(tf.maximum((1 - d), 0))
        return tf.reduce_sum(tmp + tmp2) / 2

    def __init__(self, sequence_length1, sequence_length2, channel_num, rnn_hidden_size):

        # Placeholders for input, output and dropout
        self.input_x1 = tf.placeholder(tf.float32, [None, sequence_length1, channel_num], name="input_x1")
        self.input_x2 = tf.placeholder(tf.float32, [None, sequence_length2, channel_num], name="input_x2")
        self.input_y = tf.placeholder(tf.float32, [None], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0, name="l2_loss")

        # Create a convolution + maxpool layer for each filter size
        with tf.name_scope("output"):
            self.out1 = self.BiRNN(self.input_x1, self.dropout_keep_prob, "side1", rnn_hidden_size)
            self.out2 = self.BiRNN(self.input_x2, self.dropout_keep_prob, "side2", rnn_hidden_size)
            self.distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.out1, self.out2)), 1, keep_dims=True))
            self.distance = tf.div(self.distance,
                                   tf.add(tf.sqrt(tf.reduce_sum(tf.square(self.out1), 1, keep_dims=True)),
                                          tf.sqrt(tf.reduce_sum(tf.square(self.out2), 1, keep_dims=True))))
            self.distance = tf.reshape(self.distance, [-1], name="distance")
        with tf.name_scope("loss"):
            self.loss = self.contrastive_loss(self.input_y, self.distance)
        with tf.name_scope("accuracy"):
            self.temp_sim = tf.subtract(tf.ones_like(self.distance), tf.rint(self.distance), name="temp_sim")
            correct_predictions = tf.equal(self.temp_sim, self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
