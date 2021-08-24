import os
import sys
import datetime
import tensorflow as tf
import numpy as np
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn


# Train a neural network
def nn_train(train_x1, train_x2, y_train, PARAMETERS):
    x_train = np.concatenate((train_x1, train_x2), axis=1)
    with tf.compat.v1.Graph().as_default():
        session_conf = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.compat.v1.Session(config=session_conf)
        with sess.as_default():
            if PARAMETERS.nn_type == 'AttBiRNN':
                nn = AttBiRNN(sequence_length=x_train.shape[1], num_classes=y_train.shape[1],
                              channel_num=x_train.shape[2], rnn_hidden_size=PARAMETERS.rnn_hidden_size,
                              attention_size=PARAMETERS.rnn_attention_size)
            elif PARAMETERS.nn_type == 'BiRNN':
                nn = BiRNN(sequence_length=x_train.shape[1], num_classes=y_train.shape[1],
                           channel_num=x_train.shape[2], rnn_hidden_size=PARAMETERS.rnn_hidden_size)
            elif PARAMETERS.nn_type == 'MLP':
                nn = MLP(sequence_length=x_train.shape[1], num_classes=y_train.shape[1], channel_num=x_train.shape[2],
                         hidden_size=PARAMETERS.mlp_hidden_size)
            else:
                print('unknown %s' % PARAMETERS.nn_type)
                sys.exit(0)

            # Define Training procedure
            global_step = tf.compat.v1.Variable(0, name="global_step", trainable=False)
            optimizer = tf.compat.v1.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(nn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step, name="train_op")

            # Summaries for loss and accuracy
            # loss_summary = tf.summary.scalar("loss", nn.loss)
            # acc_summary = tf.summary.scalar("accuracy", nn.accuracy)
            # train_summary_op = tf.compat.v1.summary.merge([loss_summary, acc_summary])
            # train_summary_dir = os.path.join(PARAMETERS.nn_dir, "summaries", "train")
            # train_summary_writer = tf.compat.v1.summary.FileWriter(train_summary_dir, sess.graph)

            # Checkpoint directory
            checkpoint_dir = os.path.abspath(os.path.join(PARAMETERS.nn_dir, "checkpoints"))
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=5)

            # Initialize all variables
            sess.run(tf.compat.v1.global_variables_initializer())

            def train_step(train_x_batch, train_y_batch):
                feed_dict = {
                    nn.input_x: train_x_batch,
                    nn.input_y: train_y_batch,
                    nn.dropout_keep_prob: 0.5
                }
                _, step, loss, accuracy = sess.run([train_op, global_step, nn.loss, nn.accuracy], feed_dict)
                time_str = datetime.datetime.now().isoformat()
                if step % PARAMETERS.evaluate_every == 0:
                    print("\t {}: step {}, train loss {:g}, train acc {:g}".format(time_str, step, loss, accuracy))

            # train_summary_writer.add_summary(summaries, step)

            def train_evaluate(train_x_all, train_y_all):
                feed_dict = {
                    nn.input_x: train_x_all,
                    nn.input_y: train_y_all,
                    nn.dropout_keep_prob: 0.5
                }
                loss, accuracy = sess.run([nn.loss, nn.accuracy], feed_dict)
                print("\t train loss {:g}, train acc {:g}".format(loss, accuracy))

            batches = batch_iter(list(zip(train_x1, train_x2, y_train)), PARAMETERS.num_epochs, PARAMETERS.batch_size)
            current_step = 0
            for batch in batches:
                x1_batch, x2_batch, y_batch = zip(*batch)
                x_batch = np.concatenate((x1_batch, x2_batch), axis=1)
                train_step(x_batch, y_batch)
                current_step = tf.compat.v1.train.global_step(sess, global_step)

            train_evaluate(x_train, y_train)

            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            print("\t Saved model checkpoint to {}\n".format(os.path.basename(path)))


# Predict with a trained neural network
# input: test_x1, test_x2
# output: test_p (an array, each item represents the score of one sample)
def nn_predict(test_x1, test_x2, nn_dir):
    checkpoint_dir = os.path.join(nn_dir, 'checkpoints')
    checkpoint_file = tf.compat.v1.train.latest_checkpoint(checkpoint_dir)

    graph = tf.compat.v1.Graph()
    with graph.as_default():
        session_conf = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.compat.v1.Session(config=session_conf)

        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.compat.v1.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)
            n_input_x = graph.get_operation_by_name("input_x").outputs[0]
            n_dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            n_scores = graph.get_operation_by_name("output/scores").outputs[0]
            # n_labels = graph.get_operation_by_name("output/labels").outputs[0]

            # predict
            test_x = np.concatenate((test_x1, test_x2), axis=1)
            test_score = sess.run(n_scores, {n_input_x: test_x, n_dropout_keep_prob: 1.0})

    return test_score


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


# AttBiRNN
class AttBiRNN(object):

    @staticmethod
    def attention(inputs, attention_size, return_alphas=True):
        if isinstance(inputs, tuple):
            inputs = tf.compat.v1.concat(inputs, 2)
        hidden_size = inputs.shape[2].value
        w_omega = tf.compat.v1.Variable(tf.compat.v1.random_normal([hidden_size, attention_size], stddev=0.1), name="w_omega")
        b_omega = tf.compat.v1.Variable(tf.compat.v1.random_normal([attention_size], stddev=0.1), name="b_omega")
        u_omega = tf.compat.v1.Variable(tf.compat.v1.random_normal([attention_size], stddev=0.1), name="u_omega")
        with tf.compat.v1.name_scope('v'):
            v = tf.compat.v1.tanh(tf.compat.v1.tensordot(inputs, w_omega, axes=1) + b_omega)
        vu = tf.compat.v1.tensordot(v, u_omega, axes=1, name='vu')
        alphas = tf.compat.v1.nn.softmax(vu, name='alphas')
        # output = tf.compat.v1.reduce_sum(inputs * tf.compat.v1.expand_dims(alphas, -1), 1)
        output = inputs * tf.compat.v1.expand_dims(alphas, -1)
        output = tf.compat.v1.reshape(output, [-1, output.shape[1] * output.shape[2]])
        return output if not return_alphas else output, alphas

    def __init__(self, sequence_length, num_classes, channel_num, rnn_hidden_size, attention_size):
        self.input_x = tf.compat.v1.placeholder(tf.compat.v1.float32, [None, sequence_length, channel_num], name="input_x")
        self.input_y = tf.compat.v1.placeholder(tf.compat.v1.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.compat.v1.placeholder(tf.compat.v1.float32, name="dropout_keep_prob")

        # Bidirectional RNN
        self.rnn_outputs, _ = bi_rnn(tf.compat.v1.nn.rnn_cell.GRUCell(rnn_hidden_size),
                                     tf.compat.v1.nn.rnn_cell.GRUCell(rnn_hidden_size),
                                     inputs=self.input_x, dtype=tf.compat.v1.float32)

        # Attention layer and a dropout layer
        with tf.compat.v1.name_scope('Attention_layer'):
            self.att_output, alphas = self.attention(inputs=self.rnn_outputs, attention_size=attention_size)
        with tf.compat.v1.name_scope("dropout"):
            self.att_drop = tf.compat.v1.nn.dropout(self.att_output, self.dropout_keep_prob, name="dropout")

        # FC layer
        with tf.compat.v1.name_scope("output"):
            FC_W = tf.compat.v1.get_variable("FC_W", shape=[sequence_length * rnn_hidden_size * 2, num_classes],
                                             initializer=tf.initializers.glorot_uniform())
            FC_b = tf.compat.v1.Variable(tf.compat.v1.constant(0.1, shape=[num_classes]), name="FC_b")
            self.fc_out = tf.compat.v1.nn.xw_plus_b(self.att_drop, FC_W, FC_b, name="FC_out")
            self.scores = tf.compat.v1.nn.softmax(self.fc_out, name='scores')
            self.labels = tf.compat.v1.argmax(self.scores, 1, name="labels")

        with tf.compat.v1.name_scope("loss"):
            losses = tf.compat.v1.nn.softmax_cross_entropy_with_logits(logits=self.fc_out, labels=self.input_y)
            self.loss = tf.compat.v1.reduce_mean(losses, name='loss')

        with tf.compat.v1.name_scope("accuracy"):
            correct_predictions = tf.compat.v1.equal(self.labels, tf.compat.v1.argmax(self.input_y, 1))
            self.accuracy = tf.compat.v1.reduce_mean(tf.compat.v1.cast(correct_predictions, "float"), name="accuracy")


# BiRNN
class BiRNN(object):

    def __init__(self, sequence_length, num_classes, channel_num, rnn_hidden_size):
        self.input_x = tf.compat.v1.placeholder(tf.compat.v1.float32, [None, sequence_length, channel_num], name="input_x")
        self.input_y = tf.compat.v1.placeholder(tf.compat.v1.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.compat.v1.placeholder(tf.compat.v1.float32, name="dropout_keep_prob")

        # Bidirectional RNN
        self.rnn_outputs, _ = bi_rnn(tf.compat.v1.nn.rnn_cell.GRUCell(rnn_hidden_size),
                                     tf.compat.v1.nn.rnn_cell.GRUCell(rnn_hidden_size),
                                     inputs=self.input_x, dtype=tf.compat.v1.float32)
        self.rnn_output = tf.compat.v1.concat(self.rnn_outputs, 2)
        self.rnn_output_mean = tf.compat.v1.reduce_mean(self.rnn_output, axis=1)

        # FC layer
        with tf.compat.v1.name_scope("output"):
            FC_W = tf.compat.v1.get_variable("FC_W", shape=[rnn_hidden_size * 2, num_classes],
                                             initializer=tf.initializers.glorot_uniform())
            FC_b = tf.compat.v1.Variable(tf.compat.v1.constant(0.1, shape=[num_classes]), name="FC_b")
            self.fc_out = tf.compat.v1.nn.xw_plus_b(self.rnn_output_mean, FC_W, FC_b, name="FC_out")
            self.scores = tf.compat.v1.nn.softmax(self.fc_out, name='scores')
            self.labels = tf.compat.v1.argmax(self.scores, 1, name="labels")

        with tf.compat.v1.name_scope("loss"):
            losses = tf.compat.v1.nn.softmax_cross_entropy_with_logits(logits=self.fc_out, labels=self.input_y)
            self.loss = tf.compat.v1.reduce_mean(losses, name='loss')

        with tf.compat.v1.name_scope("accuracy"):
            correct_predictions = tf.compat.v1.equal(self.labels, tf.compat.v1.argmax(self.input_y, 1))
            self.accuracy = tf.compat.v1.reduce_mean(tf.compat.v1.cast(correct_predictions, "float"), name="accuracy")


# Multiple Layer Perception
class MLP(object):

    def __init__(self, sequence_length, num_classes, channel_num, hidden_size):
        self.input_x = tf.compat.v1.placeholder(tf.compat.v1.float32, [None, sequence_length, channel_num], name="input_x")
        self.input_y = tf.compat.v1.placeholder(tf.compat.v1.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.compat.v1.placeholder(tf.compat.v1.float32, name="dropout_keep_prob")

        # FC layer
        with tf.compat.v1.name_scope("output"):
            FC_W1 = tf.compat.v1.get_variable("FC_W1", shape=[sequence_length * channel_num, hidden_size],
                                              initializer=tf.initializers.glorot_uniform())
            FC_b1 = tf.compat.v1.Variable(tf.compat.v1.constant(0.1, shape=[hidden_size]), name="FC_b1")
            self.fc_out1 = tf.compat.v1.nn.xw_plus_b(tf.compat.v1.reshape(self.input_x, [-1, sequence_length * channel_num]),
                                                     FC_W1, FC_b1, name="FC_out1")
            FC_W2 = tf.compat.v1.get_variable("FC_W2", shape=[hidden_size, num_classes],
                                              initializer=tf.initializers.glorot_uniform())
            FC_b2 = tf.compat.v1.Variable(tf.compat.v1.constant(0.1, shape=[num_classes]), name="FC_b2")
            self.fc_out2 = tf.compat.v1.nn.xw_plus_b(self.fc_out1, FC_W2, FC_b2, name="FC_out2")

            self.scores = tf.compat.v1.nn.softmax(self.fc_out2, name='scores')
            self.labels = tf.compat.v1.argmax(self.scores, 1, name="labels")

        with tf.compat.v1.name_scope("loss"):
            losses = tf.compat.v1.nn.softmax_cross_entropy_with_logits(logits=self.fc_out2, labels=self.input_y)
            self.loss = tf.compat.v1.reduce_mean(losses, name='loss')

        with tf.compat.v1.name_scope("accuracy"):
            correct_predictions = tf.compat.v1.equal(self.labels, tf.compat.v1.argmax(self.input_y, 1))
            self.accuracy = tf.compat.v1.reduce_mean(tf.compat.v1.cast(correct_predictions, "float"), name="accuracy")
