import xml.etree.ElementTree as ET
import argparse
from lib.Label import uri_prefix
"""
Given a file of scored mappings, and an OAEI reference mapping (complete gold standard) file, 
Output Precision, Recall and F1 Score
"""

parser = argparse.ArgumentParser()
parser.add_argument('--prediction_out_file', type=str, default='FMA-NCI-small/predict_score.txt')
parser.add_argument('--oaei_GS_file', type=str, default='FMA-NCI-small/oaei_FMA2NCI_UMLS_mappings_with_flagged_repairs.rdf')
parser.add_argument('--anchor_mapping_file', type=str, default='FMA-NCI-small/logmap_output/logmap_anchors.txt')
parser.add_argument('--threshold', type=float, default=0.9)
FLAGS, unparsed = parser.parse_known_args()


def read_oaei_mappings(file_name):
    tree = ET.parse(file_name)
    mappings_str = list()
    all_mappings_str = list()
    for t in tree.getroot().getchildren():
        for m in t.getchildren():
            if 'map' in m.tag:
                for c in m.getchildren():
                    mapping = list()
                    mv = '?'
                    for i, v in enumerate(c.getchildren()):
                        if i < 2:
                            for value in v.attrib.values():
                                mapping.append(uri_prefix(value).lower())
                                break
                        if i == 3:
                            mv = v.text
                    all_mappings_str.append('|'.join(mapping))
                    if not mv == '?':
                        mappings_str.append('|'.join(mapping))
    return mappings_str, all_mappings_str


if __name__ == "__main__":

    ref_mappings_str, ref_all_mappings_str = read_oaei_mappings(file_name=FLAGS.oaei_GS_file)
    ref_excluded_mappings_str = set(ref_all_mappings_str) - set(ref_mappings_str)

    anchor_mappings_str = list()
    with open(FLAGS.anchor_mapping_file) as f:
        for line in f.readlines():
            tmp = line.strip().split('|')
            anchor_mappings_str.append('%s|%s' % (uri_prefix(tmp[0]).lower(), uri_prefix(tmp[1]).lower()))

    pred_mappings_str = list()
    with open(FLAGS.prediction_out_file) as f:
        lines = f.readlines()
        for j in range(0, len(lines), 3):
            tmp = lines[j].split('|')
            if float(tmp[3]) >= FLAGS.threshold:
                pred_mappings_str.append('%s|%s' % (tmp[1].lower(), tmp[2].lower()))

    for a in anchor_mappings_str:
        if a not in pred_mappings_str:
            pred_mappings_str.append(a)

    recall_num = 0
    for s in ref_mappings_str:
        if s in pred_mappings_str:
            recall_num += 1
    R = recall_num / len(ref_mappings_str)
    precision_num = 0
    num = 0
    for s in pred_mappings_str:
        if s not in ref_excluded_mappings_str:
            if s in ref_mappings_str:
                precision_num += 1
            num += 1
    P = precision_num / num
    F1 = 2 * P * R / (P + R)
    print('Mapping #: %d, Precision: %.3f, Recall: %.3f, F1: %.3f' % (len(pred_mappings_str), P, R, F1))
