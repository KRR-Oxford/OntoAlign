import argparse
import json
import math
from owlready2 import *
from lib.Label import uri_prefix

parser = argparse.ArgumentParser()
parser.add_argument('--onto_file', type=str, default='data/foodon-merged.owl', help='data/helis_v1.00.owl')
parser.add_argument('--name_file', type=str, default='data/foodon_class_name.json', help='data/helis_class_name.json')
parser.add_argument('--path_file', type=str, default='data/foodon_all_paths.txt', help='data/helis_all_paths.txt')
FLAGS, unparsed = parser.parse_known_args()


def super_classes(c):
    supclasses = list()
    for supclass in c.is_a:
        if type(supclass) == entity.ThingClass:
            supclasses.append(supclass)
    return supclasses


# Get the class names (IRI name, and english label) for each class.
def get_class_name(o):
    c_name = dict()
    for c in o.classes():
        name = c.name
        labels = c.label.en + c.label
        names = [name, labels[0]] if len(labels) > 0 else [name, None]
        c_name[uri_prefix(uri=c.iri)] = names
    return c_name


def append_super_class(c, p):
    p.append(uri_prefix(uri=c.iri))
    supclasses = super_classes(c=c)
    if owl.Thing in supclasses or len(supclasses) == 0 or supclasses is None:
        return p
    else:
        return append_super_class(c=supclasses[0], p=p)


# Get one path to root for each class
def get_class_path(o):
    ps = []
    for c in o.classes():
        p = append_super_class(c=c, p=list())
        ps.append(p)
    return ps


# Get te maximum depth of a class to the root
def depth_max(c):
    supclasses = super_classes(c=c)
    if len(supclasses) == 0:
        return 0
    d_max = 0
    for super_c in supclasses:
        super_d = depth_max(c=super_c)
        if super_d > d_max:
            d_max = super_d
    return d_max + 1


# Get te minimum depth of a class to the root
def depth_min(c):
    supclasses = super_classes(c=c)
    if len(supclasses) == 0:
        return 0
    d_min = math.inf
    for super_c in supclasses:
        super_d = depth_min(c=super_c)
        if super_d < d_min:
            d_min = super_d
    return d_min + 1


if __name__ == "__main__":
    onto = get_ontology(FLAGS.onto_file).load()

    # To test depth_min(c) and depth_max(c), both of which are not needed in LogMap-ML
    '''
    cls = IRIS["http://purl.obolibrary.org/obo/FOODON_03413647"]
    d = depth_max(c=cls)
    d2 = depth_min(c=cls)
    '''

    print('---- extracting class names ----')
    class_name = get_class_name(o=onto)
    json.dump(class_name, open(FLAGS.name_file, 'w'))

    print('---- extract class paths ----')
    paths = get_class_path(o=onto)
    with open(FLAGS.path_file, 'w') as f:
        for path in paths:
            f.write('%s\n' % ','.join(path))

    print('---- Done! ----')
