import os
import argparse
import json
from datetime import datetime
from gensim.models import Word2Vec
from lib.Label import uri_prefix, get_label
from lib.Encoder import to_samples
from lib.Network import nn_predict
from lib.SiameseNetwork import siamese_nn_predict

parser = argparse.ArgumentParser()
parser.add_argument('--left_path_file', type=str, default='helis_foodon/helis_all_paths.txt')
parser.add_argument('--right_path_file', type=str, default='helis_foodon/foodon_all_paths.txt')
parser.add_argument('--left_class_name_file', type=str, default='helis_foodon/helis_class_name.json')
parser.add_argument('--right_class_name_file', type=str, default='helis_foodon/foodon_class_name.json')
parser.add_argument('--closest_anns_file', type=str, default='')
parser.add_argument('--candidate_file', type=str, default='logmap_overestimation.txt',
                    help='candidate mappings e.g., logmap overlapping mappings')
parser.add_argument('--prediction_out_file', type=str, default='predict_score.txt')
parser.add_argument('--class_word_size', type=int, default=14,
                    help='max. tokens in a class: 14 for HeLis/FoodOn, 6 for Conference; '
                    'it has no impact currently as word vector averaging is adopted.')
parser.add_argument('--left_path_size', type=int, default=7,
                    help='path as input: 7 for HeLis, 7 for Conference; class as input: 1')
parser.add_argument('--right_path_size', type=int, default=31,
                    help='path as input: 31 for FoodOn, 7 for Conference; class as input: 1')
parser.add_argument('--left_w2v_dir', type=str, default='word2vec_gensim',
                    help='OWL2Vec or Word2Vec of the left ontology')
parser.add_argument('--right_w2v_dir', type=str, default='word2vec_gensim',
                    help='OWL2Vec or Word2Vec of the left ontology')

parser.add_argument('--path_type', type=str, default='label', help='label, path, uri+label; '
                                                                   'set it to be consistent with train_valid.py')
parser.add_argument('--vec_type', type=str, default='word',
                    help='word, uri, word-uri; please set it to word by default;')
parser.add_argument('--keep_uri', type=str, default='no',
                    help='keep uri in the sample file or use the labels only;'
                         'set it to yes for HeLis and FoodOn; set it to no for the OAEI conference track')

parser.add_argument('--encoder_type', type=str, default='class-con',
                    help='concatenate the vectors of the classes of a path if path is used as the input;'
                         'set it to be consistent with train_valid.py')
parser.add_argument('--nn_dir', type=str, default='model_label', help='the folder of models that are trained')
parser.add_argument('--nn_type', type=str, default='SiameseMLP',
                    help='MLP, BiRNN, AttBiRNN, SiameseMLP, SiameseBiRNN, SiameseAttBiRNN')
FLAGS, unparsed = parser.parse_known_args()

setting = '%s_%s' % (FLAGS.nn_type, FLAGS.encoder_type)
FLAGS.nn_dir = os.path.join(FLAGS.nn_dir, setting)

left_paths = [line.strip().split(',') for line in open(FLAGS.left_path_file).readlines()]
right_paths = [line.strip().split(',') for line in open(FLAGS.right_path_file).readlines()]
left_names = json.load(open(FLAGS.left_class_name_file))
right_names = json.load(open(FLAGS.right_class_name_file))


mappings, mappings_n = list(), list()
with open(FLAGS.candidate_file) as f:
    for i, line in enumerate(f.readlines()):
        m = line.strip().split(', ')[1] if ', ' in line else line.strip()
        m_split = m.split('|')
        c1 = uri_prefix(uri=m_split[0])
        c2 = uri_prefix(uri=m_split[1])
        n1 = get_label(cls=c1, paths=left_paths, names=left_names, label_type='path',
                       keep_uri=(FLAGS.keep_uri == 'yes'))
        n2 = get_label(cls=c2, paths=right_paths, names=right_names, label_type='path',
                       keep_uri=(FLAGS.keep_uri == 'yes'))

        origin = 'i=%d|%s|%s' % (i + 1, c1, c2)
        name = '%s|%s' % (n1, n2)
        mappings.append(origin)
        mappings_n.append(name)

start = datetime.now()

left_wv_model = Word2Vec.load(FLAGS.left_w2v_dir)
right_wv_model = Word2Vec.load(FLAGS.right_w2v_dir)
X1, X2 = to_samples(mappings=mappings, mappings_n=mappings_n, FLAGS=FLAGS, left_wv_model=left_wv_model,
                    right_wv_model=right_wv_model)
if FLAGS.nn_type.startswith('Siamese'):
    test_distances = siamese_nn_predict(test_x1=X1, test_x2=X2, nn_dir=FLAGS.nn_dir)
    test_scores = 1 - test_distances
else:
    test_scores = nn_predict(test_x1=X1, test_x2=X2, nn_dir=FLAGS.nn_dir)
    test_scores = test_scores[:, 1]

with open(FLAGS.prediction_out_file, 'w') as f:
    for i, mapping in enumerate(mappings):
        f.write('%s|%.3f\n' % (mapping, test_scores[i]))
        f.write('%s\n' % mappings_n[i])
        f.write('\n')
print('%d mappings, all predicted' % len(mappings))
print('time cost:')
print(datetime.now()-start)
