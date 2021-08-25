import json
import re
import argparse

'''
This is to generate "named" path mappings from "URI" path mappings
The "named" path mappings enable easier human-machine interaction
'''

parser = argparse.ArgumentParser()
parser.add_argument('--path_mapping_file', type=str, default='Exp2_Distant/path_mappings.txt')
parser.add_argument('--labeled_path_mapping_file', type=str, default='Exp2_Distant/path_mappings_labeled.txt')
parser.add_argument('--left_path_file', type=str, default='data/helis_all_paths.txt')
parser.add_argument('--right_path_file', type=str, default='data/foodon_all_paths.txt')
parser.add_argument('--left_class_name_file', type=str, default='data/helis_class_name.json')
parser.add_argument('--right_class_name_file', type=str, default='data/foodon_class_name.json')
FLAGS, unparsed = parser.parse_known_args()


namespaces = ["http://www.fbk.eu/ontologies/virtualcoach#",
              "http://purl.obolibrary.org/obo/",
              "http://bioontology.org/projects/ontologies/fma/fmaOwlDlComponent_2_0#",
              "http://www.ihtsdo.org/snomed#",
              "http://www.orpha.net/ORDO/",
              "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#"]
prefixes = ["vc:", "obo:", "fma:", "snomed:", "ordo:", "nci:"]


def uri_prefix(uri):
    for i, namespace in enumerate(namespaces):
        if namespace in uri:
            return uri.replace(namespace, prefixes[i])
    return uri


def prefix_uri(ns_uri):
    for i, prefix in enumerate(prefixes):
        if prefix in ns_uri:
            return ns_uri.replace(prefix, namespaces[i])
    return ns_uri


def label_preprocess(label):
    if label is None:
        return ''
    else:
        return label.lower().replace('"', '')


def uri_name_to_string(uri_name):
    """parse the URI name (camel cases)"""
    uri_name = uri_name.replace('_', ' ').replace('-', ' ').replace('.', ' ').\
        replace('/', ' ').replace('"', ' ').replace("'", ' ')
    words = []
    for item in uri_name.split():
        matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', item)
        for m in matches:
            word = m.group(0)
            words.append(word.lower())
    return ' '.join(words)


def entity_to_string(ent, names):
    name = names[ent]
    if name[1] is None:
        uri_name = name[0]
        name_str = uri_name_to_string(uri_name=uri_name)
    else:
        label = name[1]
        name_str = label_preprocess(label=label)
    return '"%s"' % name_str


def path_to_string(path, names, keep_uri):
    names_ = list()
    for e in path:
        if keep_uri:
            names_.append('"%s"' % e)
        else:
            names_.append(entity_to_string(ent=e, names=names))
    return ','.join(names_)


def get_label(cls, paths, names, label_type, keep_uri=False):
    if label_type == 'path':
        for p in paths:
            if cls in p:
                path = p[p.index(cls):]
                return path_to_string(path=path, names=names, keep_uri=keep_uri)
    else:
        return path_to_string(path=[cls], names=names, keep_uri=keep_uri)
    return '""'


if __name__ == "__main__":
    helis_names = json.load(open(FLAGS.left_class_name_file))
    foodon_names = json.load(open(FLAGS.right_class_name_file))

    path_mappings = list()
    with open(FLAGS.path_mapping_file) as f:
        for line in f.readlines():
            mapping = line.strip().split('|')
            path_mappings.append(mapping)

    path_mappings_check = list()
    for mapping in path_mappings:
        gs_id, p1, p2 = mapping[0], mapping[1], mapping[2]
        p1_str = path_to_string(path=p1.split(','), names=helis_names, keep_uri=False)
        p2_str = path_to_string(path=p2.split(','), names=foodon_names, keep_uri=False)
        s_origin = '%s|origin|%s|%s\n' % (gs_id, p1, p2)
        s_name = '%s|name|%s|%s\n' % (gs_id, p1_str, p2_str)
        path_mappings_check.append(s_origin)
        path_mappings_check.append(s_name)
        path_mappings_check.append('\n')

    with open(FLAGS.labeled_path_mapping_file, 'w') as f:
        f.writelines(path_mappings_check)
    print('%s saved' % FLAGS.labeled_path_mapping_file)
