import csv
import numpy as np
from lib.Label import prefix_uri, uri_name_to_string
# from pattern.text.en import tokenize
from nltk.tokenize import word_tokenize


def PathEncoderWordCon(name_path, class_num, word_num, wv_model):
    # name_path.reverse()
    wv_dim = wv_model.vector_size
    name_path = name_path[0:class_num] if len(name_path) >= class_num else name_path + ['NaN'] * (
                class_num - len(name_path))

    sequence = list()
    for item in name_path:
        words = to_words(item=item)
        words = words[0:word_num] if len(words) >= word_num else words + ['NaN'] * (word_num - len(words))
        sequence = sequence + words

    e = np.zeros((len(sequence), wv_dim))
    for i, word in enumerate(sequence):
        if word == 'NaN' or word not in wv_model.wv:
            e[i, :] = np.zeros(wv_dim)
        else:
            e[i, :] = wv_model.wv[word]
    return e


def PathEncoderClassCon(name_path, class_num, wv_model):
    # name_path.reverse()
    wv_dim = wv_model.vector_size
    name_path = name_path[0:class_num] if len(name_path) >= class_num else name_path + ['NaN'] * (
               class_num - len(name_path))

    e = np.zeros((len(name_path), wv_dim))
    for i, item in enumerate(name_path):
        if item == 'NaN':
            e[i, :] = np.zeros(wv_dim)
        else:
            e[i, :] = PathEncoderWordAvg(name_path=[item], wv_model=wv_model)

    return e


def PathEncoderAvg(cls, name_path, wv_model, vec_type):
    if vec_type == 'word':
        return PathEncoderWordAvg(name_path=name_path, wv_model=wv_model)
    elif vec_type == 'uri':
        return URI_Vector(cls=cls, wv_model=wv_model)
    else:
        word_avg = PathEncoderWordAvg(name_path=name_path, wv_model=wv_model)
        uri_avg = URI_Vector(cls=cls, wv_model=wv_model)
        return np.concatenate((word_avg, uri_avg))


def URI_Vector(cls, wv_model):
    cls_uri = prefix_uri(ns_uri=cls)
    if cls_uri in wv_model.wv:
        return wv_model.wv[cls_uri]
    else:
        return np.zeros(wv_model.vector_size)


def PathEncoderWordAvg(name_path, wv_model):
    wv_dim = wv_model.vector_size
    num, v = 0, np.zeros(wv_dim)
    for item in name_path:
        for word in to_words(item=item):
            if word in wv_model.wv:
                num += 1
                v += wv_model.wv[word]
    avg = (v / num) if num > 0 else v
    return avg


def to_words(item):
    if item.startswith('http://'):
        if '#' in item:
            uri_name = item.split('#')[1]
        else:
            uri_name = item.split('/')[-1]
        words_str = uri_name_to_string(uri_name=uri_name)
        words = words_str.split(' ')
    else:
        item = item.replace('_', ' ').replace('-', ' ').replace('.', ' ').replace('/', ' '). \
            replace('"', ' ').replace("'", ' ').replace('\\', ' ').replace('(', ' ').replace(')', ' ')
        tokenized_line = ' '.join(word_tokenize(item))
        # words = [word for word in tokenized_line.lower().split() if word.isalpha()]
        words = [word for word in tokenized_line.lower().split()]
    return words


# for training, with output Y
def load_samples(file_name, FLAGS, left_wv_model, right_wv_model):
    if FLAGS.path_type == 'label':
        FLAGS.left_path_size = 1
        FLAGS.right_path_size = 1
    if FLAGS.path_type == 'uri+label':
        FLAGS.left_path_size = 3
        FLAGS.right_path_size = 3

    left_wv_dim = left_wv_model.vector_size
    right_wv_dim = right_wv_model.vector_size
    lines = open(file_name).readlines()
    num = int(len(lines)/3)
    if FLAGS.encoder_type == 'word-con':
        X1 = np.zeros((num, FLAGS.left_path_size * FLAGS.class_word_size, left_wv_dim))
        X2 = np.zeros((num, FLAGS.right_path_size * FLAGS.class_word_size, right_wv_dim))
    elif FLAGS.encoder_type == 'class-con':
        X1 = np.zeros((num, FLAGS.left_path_size, left_wv_dim))
        X2 = np.zeros((num, FLAGS.right_path_size, right_wv_dim))
    else:
        if FLAGS.vec_type == 'word-uri':
            X1 = np.zeros((num, 1, left_wv_dim * 2))
            X2 = np.zeros((num, 1, right_wv_dim * 2))
        else:
            X1 = np.zeros((num, 1, left_wv_dim))
            X2 = np.zeros((num, 1, right_wv_dim))
    Y = np.zeros((num, 2))

    for i in range(0, len(lines), 3):
        name_mapping = lines[i+1]
        tmp = name_mapping.split('|')
        p1 = [x for x in list(csv.reader([tmp[2]], delimiter=',', quotechar='"'))[0]]
        p2 = [x for x in list(csv.reader([tmp[3]], delimiter=',', quotechar='"'))[0]]
        mapping = lines[i].strip().split('|')
        left_c, right_c = mapping[2], mapping[3]

        if FLAGS.path_type == 'label':
            p1 = [p1[0]]
            p2 = [p2[0]]
        if FLAGS.path_type == 'uri+label':
            p1 = [left_c.split(':')[1]] + p1
            p2 = [right_c.split(':')[1]] + p2

        j = int(i/3)
        if FLAGS.encoder_type == 'word-con':
            X1[j] = PathEncoderWordCon(name_path=p1, wv_model=left_wv_model, class_num=FLAGS.left_path_size, word_num=FLAGS.class_word_size)
            X2[j] = PathEncoderWordCon(name_path=p2, wv_model=right_wv_model, class_num=FLAGS.right_path_size, word_num=FLAGS.class_word_size)
        elif FLAGS.encoder_type == 'class-con':
            X1[j] = PathEncoderClassCon(name_path=p1, wv_model=left_wv_model, class_num=FLAGS.left_path_size)
            X2[j] = PathEncoderClassCon(name_path=p2, wv_model=right_wv_model, class_num=FLAGS.right_path_size)
        else:
            X1[j, 0] = PathEncoderAvg(cls=left_c, name_path=p1, wv_model=left_wv_model, vec_type=FLAGS.vec_type)
            X2[j, 0] = PathEncoderAvg(cls=right_c, name_path=p2, wv_model=right_wv_model, vec_type=FLAGS.vec_type)
        Y[j] = np.array([1.0, 0.0]) if tmp[0].startswith('neg') else np.array([0.0, 1.0])

    return X1, X2, Y, num


# for prediction (without output Y)
def to_samples(mappings, mappings_n, FLAGS, left_wv_model, right_wv_model):
    if FLAGS.path_type == 'label':
        FLAGS.left_path_size = 1
        FLAGS.right_path_size = 1
    if FLAGS.path_type == 'uri+label':
        FLAGS.left_path_size = 3
        FLAGS.right_path_size = 3

    left_wv_dim = left_wv_model.vector_size
    right_wv_dim = right_wv_model.vector_size
    num = len(mappings_n)
    if FLAGS.encoder_type == 'word-con':
        X1 = np.zeros((num, FLAGS.left_path_size * FLAGS.class_word_size, left_wv_dim))
        X2 = np.zeros((num, FLAGS.right_path_size * FLAGS.class_word_size, right_wv_dim))
    elif FLAGS.encoder_type == 'class-con':
        X1 = np.zeros((num, FLAGS.left_path_size, left_wv_dim))
        X2 = np.zeros((num, FLAGS.right_path_size, right_wv_dim))
    else:
        if FLAGS.vec_type == 'word-uri':
            X1 = np.zeros((num, 1, left_wv_dim * 2))
            X2 = np.zeros((num, 1, right_wv_dim * 2))
        else:
            X1 = np.zeros((num, 1, left_wv_dim))
            X2 = np.zeros((num, 1, right_wv_dim))

    for i in range(num):
        tmp = mappings_n[i].split('|')
        p1 = [x for x in list(csv.reader([tmp[0]], delimiter=',', quotechar='"'))[0]]
        p2 = [x for x in list(csv.reader([tmp[1]], delimiter=',', quotechar='"'))[0]]

        tmp = mappings[i].split('|')
        left_c, right_c = tmp[1], tmp[2]

        if FLAGS.path_type == 'label':
            p1 = p1[0:1]
            p2 = p2[0:1]
        if FLAGS.path_type == 'uri+label':
            p1 = [left_c.split(':')[1]] + p1
            p2 = [right_c.split(':')[1]] + p2

        if FLAGS.encoder_type == 'word-con':
            X1[i] = PathEncoderWordCon(name_path=p1, wv_model=left_wv_model, class_num=FLAGS.left_path_size, word_num=FLAGS.class_word_size)
            X2[i] = PathEncoderWordCon(name_path=p2, wv_model=right_wv_model, class_num=FLAGS.right_path_size, word_num=FLAGS.class_word_size)
        elif FLAGS.encoder_type == 'class-con':
            X1[i] = PathEncoderClassCon(name_path=p1, wv_model=left_wv_model, class_num=FLAGS.left_path_size)
            X2[i] = PathEncoderClassCon(name_path=p2, wv_model=right_wv_model, class_num=FLAGS.right_path_size)
        else:
            X1[i, 0] = PathEncoderAvg(cls=left_c, name_path=p1, wv_model=left_wv_model, vec_type=FLAGS.vec_type)
            X2[i, 0] = PathEncoderAvg(cls=right_c, name_path=p2, wv_model=right_wv_model, vec_type=FLAGS.vec_type)

    return X1, X2
