import random
import argparse
import json
import sys
import xml.etree.ElementTree as ET
from lib.Label import uri_prefix, get_label

'''
This is to generate train and valid class mappings (samples) from the anchor class mappings
Cross-ontology class disjointness constraints (i.e., branch conflicts) are used to filter out low quality anchors
Each sample can be a pair classes or their associated paths
'''
parser = argparse.ArgumentParser()
parser.add_argument('--anchor_mapping_file', type=str, default='logmap_anchors.txt')
parser.add_argument('--train_file', type=str, default='mappings_train.txt')
parser.add_argument('--valid_file', type=str, default='mappings_valid.txt')
parser.add_argument('--train_rate', type=float, default=1.0,
                    help='it can be set to 1.0 to use all the seeds as the training set and 20% of them as the validation set;'
                         'or a float smaller than 1.0, where train_rate of all the samples are used as the training set and the remaining are used as the validation set')
parser.add_argument('--sample_duplicate', type=int, default=2)
parser.add_argument('--left_path_file', type=str, default='data/helis_all_paths.txt',
                    help='all the paths that are pre-extracted')
parser.add_argument('--right_path_file', type=str, default='data/foodon_all_paths.txt',
                    help='all the paths that are pre-extracted')
parser.add_argument('--left_class_name_file', type=str, default='data/helis_class_name.json',
                    help='pre-extracted the class names (rdfs:label and URI name) of each class')
parser.add_argument('--right_class_name_file', type=str, default='data/foodon_class_name.json',
                    help='pre-extracted the class names (rdfs:label and URI name) of each class')
parser.add_argument('--keep_uri', type=str, default='no',
                    help='keep uri in the sample or use the path labels; '
                         'set it to yes for the OAEI conference track; set it to no for HeLis and FoodOn')
parser.add_argument('--anchor_branch_conflict', type=str, default='yes')
parser.add_argument('--generate_negative_sample', type=str, default='yes')

parser.add_argument('--anchor_GS', type=str, default='no', help='used for debug; set it to no by default')
parser.add_argument('--GS_file', type=str, default='', help='used for debug; set it to empty by default')
FLAGS, unparsed = parser.parse_known_args()

# class disjointness constraints for HeLis and FoodOn
branch_conflicts = [
    ['","nutrient"', '"food product type","material entity","independent continuant","continuant","entity"'],
    ['"basic food","food"',
     '"food source","environmental material","fiat object part","material entity","independent continuant","continuant","entity"'],
    ['"basic food","food"', '"organism","material entity"'],
    ['"basic food","food"', '"chemical entity","material entity"'],
]


def read_oaei_mappings(file_name):
    tree = ET.parse(file_name)
    mappings_str = list()
    for t in tree.getroot().getchildren():
        for m in t.getchildren():
            if 'map' in m.tag:
                for c in m.getchildren():
                    mapping = list()
                    for i, v in enumerate(c.getchildren()):
                        if i < 2:
                            for value in v.attrib.values():
                                mapping.append(value)
                                break
                    mappings_str.append('|'.join(mapping))
    return mappings_str


def violate_branch_conflict(p1_str, p2_str):
    for conflict in branch_conflicts:
        if conflict[0] in p1_str and conflict[1] in p2_str:
            return True
    return False


if FLAGS.anchor_GS == 'yes':
    if FLAGS.GS_file.endswith('.rdf'):
        GS_mappings_str = read_oaei_mappings(file_name=FLAGS.GS_file)
    elif FLAGS.GS_file.endswith('.txt'):
        GS_mappings_str = [line.strip() for line in open(FLAGS.GS_file).readlines()]
    else:
        print('%s: unsupported file format' % FLAGS.GS_file)
        sys.exit(0)


def violate_rules(p1_str, p2_str, c1_uri, c2_uri):
    violated = False
    if FLAGS.anchor_branch_conflict == 'yes':
        violated = violate_branch_conflict(p1_str=p1_str, p2_str=p2_str)
    if FLAGS.anchor_GS == 'yes':
        if '%s|%s' % (c1_uri, c2_uri) not in GS_mappings_str:
            violated = True
    return violated


def negative_sampling(pos_mappings, left_paths, right_paths, left_names, right_names):
    neg_mappings = list()
    for mapping in pos_mappings:
        tmp = mapping[0].split('|')
        m_id, c1, c2 = tmp[0], tmp[2], tmp[3]

        neg_c2 = random.sample(right_classes - {c2}, 1)[0]
        neg_n2 = get_label(cls=neg_c2, paths=right_paths, names=right_names, label_type='path',
                           keep_uri=(FLAGS.keep_uri == 'yes'))
        n1 = get_label(cls=c1, paths=left_paths, names=left_names, label_type='path',
                       keep_uri=(FLAGS.keep_uri == 'yes'))
        if not n1 == '""' and not neg_n2 == '""':
            origin = 'neg-%s-f|origin|%s|%s' % (m_id, c1, neg_c2)
            name = 'neg-%s-f|name|%s|%s' % (m_id, n1, neg_n2)
            neg_mappings.append([origin, name])

        neg_c1 = random.sample(left_classes - {c1}, 1)[0]
        neg_n1 = get_label(cls=neg_c1, paths=left_paths, names=left_names, label_type='path',
                           keep_uri=(FLAGS.keep_uri == 'yes'))
        n2 = get_label(cls=c2, paths=right_paths, names=right_names, label_type='path',
                       keep_uri=(FLAGS.keep_uri == 'yes'))
        if not neg_n1 == '""' and not n2 == '""':
            origin = 'neg-%s-h|origin|%s|%s' % (m_id, neg_c1, c2)
            name = 'neg-%s-h|name|%s|%s' % (m_id, neg_n1, n2)
            neg_mappings.append([origin, name])
    return neg_mappings


def save_mappings(mappings2, file_name):
    if len(mappings2) > 0:
        with open(file_name, 'w') as f2:
            for mapping2 in mappings2:
                f2.write(mapping2[0] + '\n')
                f2.write(mapping2[1] + '\n')
                f2.write('\n')


if __name__ == "__main__":
    left_names = json.load(open(FLAGS.left_class_name_file))
    right_names = json.load(open(FLAGS.right_class_name_file))
    left_classes = set(left_names.keys())
    right_classes = set(right_names.keys())
    left_paths = [line.strip().split(',') for line in open(FLAGS.left_path_file).readlines()]
    right_paths = [line.strip().split(',') for line in open(FLAGS.right_path_file).readlines()]

    mappings = list()
    rule_violated_mappings = list()
    with open(FLAGS.anchor_mapping_file) as f:
        for i, line in enumerate(f.readlines()):
            tmp = line.strip().split(', ')[1] if ', ' in line else line.strip()
            tmp2 = tmp.split('|')
            c1 = uri_prefix(uri=tmp2[0])
            c2 = uri_prefix(uri=tmp2[1])
            n1 = get_label(cls=c1, paths=left_paths, names=left_names, label_type='path',
                           keep_uri=(FLAGS.keep_uri == 'yes'))
            n2 = get_label(cls=c2, paths=right_paths, names=right_names, label_type='path',
                           keep_uri=(FLAGS.keep_uri == 'yes'))

            if not n1 == '""' and not n2 == '""':
                if violate_rules(p1_str=n1, p2_str=n2, c1_uri=tmp2[0], c2_uri=tmp2[1]):
                    origin = 'neg-%d|origin|%s|%s' % (i + 1, c1, c2)
                    name = 'neg-%d|name|%s|%s' % (i + 1, n1, n2)
                    rule_violated_mappings.append([origin, name])
                else:
                    origin = '%d|origin|%s|%s' % (i + 1, c1, c2)
                    name = '%d|name|%s|%s' % (i + 1, n1, n2)
                    mappings.append([origin, name])
    print('%d anchors in total, %d anchors violate the rules' % (i + 1, len(rule_violated_mappings)))

    train_num = round(len(mappings) * FLAGS.train_rate)
    random.shuffle(mappings)
    train_mappings = mappings[0:train_num]
    if FLAGS.train_rate < 1.0:
        valid_mappings = mappings[train_num:]
    else:
        valid_mappings = mappings[round(len(mappings) * 0.8):]
    print('positive -- train: %d, valid: %d' % (train_num, len(mappings) - train_num))

    random.shuffle(rule_violated_mappings)
    train_rv_num = round(len(rule_violated_mappings) * FLAGS.train_rate)
    train_rv_mappings = rule_violated_mappings[0:train_rv_num]
    if FLAGS.train_rate < 1.0:
        valid_rv_mappings = rule_violated_mappings[train_rv_num:]
    else:
        valid_rv_mappings = rule_violated_mappings[round(len(rule_violated_mappings) * 0.8):]

    if FLAGS.generate_negative_sample == 'yes':
        train_mappings = train_mappings * FLAGS.sample_duplicate + negative_sampling(pos_mappings=train_mappings,
                                                                                     left_paths=left_paths,
                                                                                     right_paths=right_paths,
                                                                                     left_names=left_names,
                                                                                     right_names=right_names) + train_rv_mappings * FLAGS.sample_duplicate
        valid_mappings = valid_mappings * FLAGS.sample_duplicate + negative_sampling(pos_mappings=valid_mappings,
                                                                                     left_paths=left_paths,
                                                                                     right_paths=right_paths,
                                                                                     left_names=left_names,
                                                                                     right_names=right_names) + valid_rv_mappings * FLAGS.sample_duplicate
    else:
        train_mappings = train_mappings + train_rv_mappings
        valid_mappings = valid_mappings + valid_rv_mappings

    save_mappings(mappings2=train_mappings, file_name=FLAGS.train_file)
    save_mappings(mappings2=valid_mappings, file_name=FLAGS.valid_file)

    print('All -- train: %d, valid: %d' % (len(train_mappings), len(valid_mappings)))
    print('Done')
