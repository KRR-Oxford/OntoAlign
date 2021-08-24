import argparse
import os
import numpy as np
from datetime import datetime
from gensim.models import Word2Vec
from lib.Encoder import load_samples
from lib.Network import nn_train, nn_predict
from lib.SiameseNetwork import siamese_nn_predict, siamese_nn_train
from lib.Evaluator import threshold_searching

parser = argparse.ArgumentParser()
parser.add_argument('--train_path_file', type=str, default='mappings_train.txt')
parser.add_argument('--valid_path_file', type=str, default='mappings_valid.txt')
parser.add_argument('--class_word_size', type=int, default=14,
                    help='max. tokens in a class: 14 for HeLis/FoodOn, 6 for Conference; '
                         'it has no impact currently as word vector averaging is adopted.')
parser.add_argument('--left_path_size', type=int, default=7,
                    help='path as input: 7 for HeLis, 7 for Conference; class as input (--path_type=label): 1')
parser.add_argument('--right_path_size', type=int, default=31,
                    help='path as input: 31 for FoodOn, 7 for Conference; class as input (--path_type=label): 1')

parser.add_argument('--left_w2v_dir', type=str, default='word2vec_gensim',
                    help='OWL2Vec or Word2Vec of the left ontology')
parser.add_argument('--right_w2v_dir', type=str, default='word2vec_gensim',
                    help='OWL2Vec or Word2Vec of the right ontology')

parser.add_argument('--vec_type', type=str, default='word',
                    help='word, uri, word-uri; please set it to word by default')

parser.add_argument('--path_type', type=str, default='label',
                    help='three settings: label, path, uri+label;'
                         'label: the class embedding as input; '
                         'path: the path embedding as input'
                         'uri+label: the uri name and label of the class')

parser.add_argument('--nn_base_dir', type=str, default='model_label/', help='the folder for the output models')
parser.add_argument('--rnn_hidden_size', type=int, default=200)
parser.add_argument('--rnn_attention_size', type=int, default=50)
parser.add_argument('--mlp_hidden_size', type=int, default=200)
parser.add_argument('--num_epochs', type=int, default=14)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--evaluate_every', type=int, default=100, help='Evaluate model after this many steps')
FLAGS, unparsed = parser.parse_known_args()

# candidate setting: class-con -- concatenate the vectors of the classes of a path
#                    avg -- average the vectors of the classes of a path
# encoder_types = ['avg', 'class-con', 'word-con']
encoder_types = ['class-con']

# candidate models that have been implemented
# SiameseBiRNNN has an on and off bug nan loss issue not addressed
# nn_types = ['MLP', 'BiRNN', 'AttBiRNN', 'SiameseMLP', 'SiameseAttBiRNN']
nn_types = ['MLP', 'SiameseMLP']

start = datetime.now()

left_wv_model = Word2Vec.load(FLAGS.left_w2v_dir)
right_wv_model = Word2Vec.load(FLAGS.right_w2v_dir)


def train(X1, X2, Y):
    if FLAGS.nn_type.startswith('Siamese'):
        siamese_nn_train(train_x1=X1, train_x2=X2, y_train=Y, PARAMETERS=FLAGS)
    else:
        nn_train(train_x1=X1, train_x2=X2, y_train=Y, PARAMETERS=FLAGS)


def valid(X1, X2, Y):
    if FLAGS.nn_type.startswith('Siamese'):
        valid_distances = siamese_nn_predict(test_x1=X1, test_x2=X2, nn_dir=FLAGS.nn_dir)
        valid_scores = 1 - valid_distances
        max_alpha, max_valid_f1, max_valid_p, max_valid_r, max_valid_acc = threshold_searching(Y=Y[:, 1],
                                                                                               scores=valid_scores,
                                                                                               num=valid_num)
    else:
        valid_scores = nn_predict(test_x1=X1, test_x2=X2, nn_dir=FLAGS.nn_dir)
        valid_scores = valid_scores[:, 1]
        max_alpha, max_valid_f1, max_valid_p, max_valid_r, max_valid_acc = threshold_searching(Y=Y[:, 1],
                                                                                               scores=valid_scores,
                                                                                               num=valid_num)
    return max_alpha, max_valid_f1, max_valid_p, max_valid_r, max_valid_acc


for encoder_type in encoder_types:
    FLAGS.encoder_type = encoder_type
    train_X1, train_X2, train_Y, train_num = load_samples(file_name=FLAGS.train_path_file, FLAGS=FLAGS,
                                                          left_wv_model=left_wv_model,
                                                          right_wv_model=right_wv_model)
    shuffle_indices = np.random.permutation(np.arange(train_num))
    train_X1, train_X2, train_Y = train_X1[shuffle_indices], train_X2[shuffle_indices], train_Y[shuffle_indices]
    valid_X1, valid_X2, valid_Y, valid_num = load_samples(file_name=FLAGS.valid_path_file, FLAGS=FLAGS,
                                                          left_wv_model=left_wv_model,
                                                          right_wv_model=right_wv_model)
    for nn_type in nn_types:
        setting = '%s_%s' % (nn_type, encoder_type)
        FLAGS.nn_type = nn_type
        FLAGS.nn_dir = os.path.join(FLAGS.nn_base_dir, setting)
        print('\n------ %s START ------' % setting)
        train(X1=train_X1, X2=train_X2, Y=train_Y)
        threshold, f1, p, r, acc = valid(X1=valid_X1, X2=valid_X2, Y=valid_Y)
        print(
            '\n ##### best setting: %s, threshold: %.2f, precision: %.3f, recall: %.3f, f1: %.3f, acc: %.3f ##### \n' % (
                setting, threshold, p, r, f1, acc))

print('done')
print('time cost:')
print(datetime.now()-start)
