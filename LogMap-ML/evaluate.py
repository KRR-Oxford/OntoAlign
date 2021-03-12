import argparse
import random
import sample
from lib.Label import uri_prefix

'''
Evaluation for HeLis and FoodOn
Recall: calculate the recall w.r.t. the GS mappings

For Precision, this program will sample and save a number of random predicted class mappings for manual annotation in --sample_to_annotate_file
In manual annotation, just add 'true' or 'false' in the end of each mapping.
An example of the annotated mapping:

vc:Tryptophan|obo:CHEBI_27897|0.892|true
"tryptophan","polar aminoacid","aminoacid","nutrient"|"tryptophan","aminoalkylindole","indoles","benzopyrrole","organonitrogen heterocyclic compound","organonitrogen compound","heteroorganic entity","organic molecular entity","carbon group molecular entity","p-block molecular entity","main group molecular entity","molecular entity","chemical entity","material entity","independent continuant","continuant","entity"

The first line is the mappings with the predicted score, where "true" is the annotation; 
the second line is the paths of the two to-be-aligned classes (the names are shown for easy human annotation).
'''

parser = argparse.ArgumentParser()
parser.add_argument('--left_path_file', type=str, default='helis_foodon/helis_all_paths.txt')
parser.add_argument('--right_path_file', type=str, default='helis_foodon/foodon_all_paths.txt')
parser.add_argument('--anchor_file', type=str, default='logmap_anchors.txt')
parser.add_argument('--prediction_out_file', type=str, default='predict_score.txt')
parser.add_argument('--GS_file', type=str, default='helis_foodon/GS_mappings_path_checked.txt')
parser.add_argument('--threshold', type=float, default=0.44)
parser.add_argument('--prediction_sample_num', type=int, default=50)
parser.add_argument('--sample_to_annotate_file', type=str, default='LogMap-ML_samples_a.txt')
parser.add_argument('--output_mapping_file', type=str, default='LogMap-ML_output_mappings.txt')
FLAGS, unparsed = parser.parse_known_args()

GS = set([line.strip() for line in open(FLAGS.GS_file).readlines()])
print('GS #: %d' % len(GS))


def cal_recall(mappings_dict):
    pred_n = 0
    for left_c in mappings_dict:
        pred_n += len(mappings_dict[left_c])
    print('Predictions #: %d' % pred_n)

    recall_num = 0
    for gs in GS:
        tmp = gs.split('|')
        gs_left_class, gs_right_class = tmp[0], tmp[1]
        if gs_left_class in mappings_dict and gs_right_class in mappings_dict[gs_left_class]:
            recall_num += 1
    recall = recall_num / len(GS)
    print('Recall w.r.t. GS: %.3f' % recall)


def read_logmap_mapping(filename):
    mappings_dict = dict()
    with open(filename) as f:
        for line in f.readlines():
            m = line.strip().split('|')
            left_c = uri_prefix(m[0])
            right_c = uri_prefix(m[1])
            if left_c in mappings_dict:
                mappings_dict[left_c].append(right_c)
            else:
                mappings_dict[left_c] = [right_c]
    return mappings_dict


def sample_save_mapping(filename, mappings_dict):
    mappings = list()
    for left_c in mappings_dict:
        for right_c in mappings_dict[left_c]:
            mappings.append('%s|%s' % (left_c, right_c))
    mappings_sample = random.sample(mappings, FLAGS.prediction_sample_num)
    with open(filename, 'w') as f:
        for mapping in mappings_sample:
            f.write('%s|%.3f\n' % (mapping, predict_mapping_score[mapping]))
            f.write('%s\n' % mapping_label[mapping])
            f.write('\n')


def save_mapping(filename, mapping_dict):
    with open(filename, 'w') as f:
        for left_c in mapping_dict:
            for right_c in mapping_dict[left_c]:
                f.write('%s|%s\n' % (left_c, right_c))


'''
get the mappings (filtered by the threshold)
'''
predict_left_right = dict()
predict_right_left = dict()
predict_mapping_score = dict()
mapping_label = dict()
with open(FLAGS.prediction_out_file) as f:
    lines = f.readlines()
    for j in range(0, len(lines), 3):
        tmp = lines[j].strip().split('|')
        left_c, right_c, score = tmp[1], tmp[2], float(tmp[3])
        mapping = '%s|%s' % (left_c, right_c)
        mapping_label[mapping] = lines[j + 1].strip()
        predict_mapping_score[mapping] = score

        if score >= FLAGS.threshold:
            if left_c in predict_left_right:
                predict_left_right[left_c].append(right_c)
            else:
                predict_left_right[left_c] = [right_c]
            if right_c in predict_right_left:
                predict_right_left[right_c].append(left_c)
            else:
                predict_right_left[right_c] = [left_c]
print('\npredict results (filtered by threshold %.2f):' % FLAGS.threshold)
cal_recall(mappings_dict=predict_left_right)

'''
filter by branch conflicts (class disjointness constraints)
'''
predict_left_right2 = dict()
with open(FLAGS.prediction_out_file) as f:
    lines = f.readlines()
    for j in range(0, len(lines), 3):
        tmp = lines[j].split('|')
        mapping_n = lines[j + 1].split('|')
        left_c, right_c = tmp[1], tmp[2]
        if float(tmp[3]) >= FLAGS.threshold:
            if not sample.violate_branch_conflict(p1_str=mapping_n[0], p2_str=mapping_n[1]):
                if left_c in predict_left_right2:
                    predict_left_right2[left_c].append(right_c)
                else:
                    predict_left_right2[left_c] = [right_c]
print('\npredict results (filtered by threshold and class disjointness constraints):')
cal_recall(mappings_dict=predict_left_right2)


def load_class_pairs_within_path(filename):
    pairs = set()
    for line in open(filename).readlines():
        path = line.strip().split(',')
        for c1 in path:
            for c2 in path:
                if not c1 == c2:
                    pairs.add('%s|%s' % (c1, c2))
                    pairs.add('%s|%s' % (c2, c1))
    return pairs


'''
filter by path-based logical assessment
'''
left_pairs = load_class_pairs_within_path(filename=FLAGS.left_path_file)
right_pairs = load_class_pairs_within_path(filename=FLAGS.right_path_file)

discarded_mappings = set()
for right_c in predict_right_left:
    left_classes = predict_right_left[right_c]
    for i in range(0, len(left_classes)):
        for j in range(i + 1, len(left_classes)):
            if '%s|%s' % (left_classes[i], left_classes[j]) in left_pairs:
                m1 = '%s|%s' % (left_classes[i], right_c)
                m2 = '%s|%s' % (left_classes[j], right_c)
                if predict_mapping_score[m1] < predict_mapping_score[m2]:
                    discarded_mappings.add(m1)
                if predict_mapping_score[m2] < predict_mapping_score[m1]:
                    discarded_mappings.add(m2)

for left_c in predict_left_right:
    right_classes = predict_left_right[left_c]
    for i in range(0, len(right_classes)):
        for j in range(i + 1, len(right_classes)):
            if '%s|%s' % (right_classes[i], right_classes[j]) in right_pairs:
                m1 = '%s|%s' % (left_c, right_classes[i])
                m2 = '%s|%s' % (left_c, right_classes[j])
                if predict_mapping_score[m1] < predict_mapping_score[m2]:
                    discarded_mappings.add(m1)
                if predict_mapping_score[m2] < predict_mapping_score[m1]:
                    discarded_mappings.add(m2)

predict_left_right_path = dict()
for left_c in predict_left_right:
    for right_c in predict_left_right[left_c]:
        if '%s|%s' % (left_c, right_c) not in discarded_mappings:
            if left_c in predict_left_right_path:
                predict_left_right_path[left_c].append(right_c)
            else:
                predict_left_right_path[left_c] = [right_c]
print('\npredict results after filtering by path-based logical assessment (without ensemble):')
cal_recall(mappings_dict=predict_left_right_path)
# save_mapping(filename='Exp2_Distant/pred_oe_d12p_cls_SMLP_path.txt', mapping_dict=predict_left_right_path)

'''
hybrid results of combing the predictions and the logmap anchors
'''
logmap_anchors = read_logmap_mapping(filename=FLAGS.anchor_file)
hybrid_left_right = dict()
for left_c in logmap_anchors:
    for right_c in logmap_anchors[left_c]:
        mapping = '%s|%s' % (left_c, right_c)
        label_split = mapping_label[mapping].split('|')
        if not sample.violate_branch_conflict(p1_str=label_split[0], p2_str=label_split[1]):
            if left_c in hybrid_left_right:
                hybrid_left_right[left_c].append(right_c)
            else:
                hybrid_left_right[left_c] = [right_c]

for left_c in predict_left_right_path:
    if left_c not in hybrid_left_right:
        hybrid_left_right[left_c] = predict_left_right_path[left_c]
    else:
        for c in predict_left_right_path[left_c]:
            if c not in hybrid_left_right[left_c]:
                hybrid_left_right[left_c].append(c)
print('\nresults with ensemble:')
cal_recall(mappings_dict=hybrid_left_right)

print('sample and save mappings')
sample_save_mapping(filename=FLAGS.sample_to_annotate_file, mappings_dict=hybrid_left_right)
save_mapping(filename=FLAGS.output_mapping_file, mapping_dict=hybrid_left_right)

print('all done')
