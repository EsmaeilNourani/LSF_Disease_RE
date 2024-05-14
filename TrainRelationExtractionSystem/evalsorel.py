#!/usr/bin/env python3

# Evaluate relation annotations in standoff data.

import sys
import os

from collections import defaultdict
from logging import error


# Relation types to treat as symmetric
SYMMETRIC_RELATION_TYPES = set([
    'Complex_formation'
])


def argparser():
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('-r', '--relations', metavar='TYPE[,TYPE[...]]',
                    default=None, help='restrict to given relation types')
    ap.add_argument('-e', '--entities', metavar='TYPE[,TYPE[...]]',
                    default=None, help='restrict to given entity types')
    ap.add_argument('-m', '--allow-missing', default=False,
                    action='store_true',
                    help='allow gold files lacking a corresponding test file.')
    ap.add_argument('-v', '--verbose', default=False, action='store_true',
                    help='verbose output')
    ap.add_argument('gold', help='directory')
    ap.add_argument('pred', help='directory')
    return ap


########## annotation ##########


class FormatError(Exception):
    pass


class SpanDeleted(Exception):
    pass


class Annotation(object):
    def __init__(self, id_, type_):
        self.id_ = id_
        self.type_ = type_
        self.matched = False

    def resolve_references(self, ann_by_id):
        pass

    def resolve_equivs(self, equiv_sets):
        pass


class Textbound(Annotation):
    def __init__(self, id_, type_, offsets, text):
        Annotation.__init__(self, id_, type_)
        self.text = text
        self.offsets = []
        for start_end in offsets.split(';'):
            start, end = start_end.split()
            self.offsets.append((int(start), int(end)))


class XMLElement(Textbound):
    def __init__(self, id_, type_, offsets, text, attributes):
        Textbound.__init__(self, id_, type_, offsets, text)
        self.attributes = attributes


class ArgAnnotation(Annotation):
    def __init__(self, id_, type_, args):
        Annotation.__init__(self, id_, type_)
        self.arg_ids = []
        for arg in args:
            role, id_ = arg.split(':')
            self.arg_ids.append((role, id_))
        self.orig_arg_ids = self.arg_ids.copy()
        self.args = None

    def roles(self):
        return set(role for role, ann in self.arg_ids)

    def targets(self):
        return set(i for r, i in self.arg_ids)
    
    def targets_by_role(self, role):
        return set(i for r, i in self.arg_ids if r == role)

    def target_annotations(self):
        return set(a for r, a in self.args)

    def resolve_references(self, ann_by_id):
        self.args = [(role, ann_by_id[id_]) for role, id_ in self.arg_ids]

    def resolve_equivs(self, equiv_sets):
        # Map argument IDs to representative in equiv set
        self.arg_ids = []
        for role, id_ in self.orig_arg_ids:
            for e in equiv_sets:
                if id_ in e:
                    id_ = sorted(e)[0]
                    break
            self.arg_ids.append((role, id_))


class Relation(ArgAnnotation):
    def __init__(self, id_, type_, arg_ids):
        ArgAnnotation.__init__(self, id_, type_, arg_ids)

    def __str__(self):
        arg_str = ' '.join('{}:{}'.format(r, i) for r, i in self.orig_arg_ids)
        return '{}\t{} {}'.format(self.id_, self.type_, arg_str)

        
class Event(ArgAnnotation):
    def __init__(self, id_, type_, trigger, arg_ids):
        ArgAnnotation.__init__(self, id_, type_, arg_ids)
        self.trigger = trigger

    def resolve_references(self, ann_by_id):
        raise NotImplementedError()    # TODO

    def resolve_equivs(self, equiv_sets):
        raise NotImplementedError()


class Attribute(Annotation):
    def __init__(self, id_, type_, target_id, value):
        Annotation.__init__(self, id_, type_)
        self.target_id = target_id
        self.value = value
        self.target = None

    def resolve_references(self, ann_by_id):
        self.target = ann_by_id[self.target_id]


class Normalization(Annotation):
    def __init__(self, id_, type_, target_id, ref, reftext):
        Annotation.__init__(self, id_, type_)
        self.target_id = target_id
        self.ref = ref
        self.reftext = reftext
        self.target = None

    def resolve_references(self, ann_by_id):
        self.target = ann_by_id[self.target_id]


class Equiv(Annotation):
    def __init__(self, id_, type_, target_ids):
        Annotation.__init__(self, id_, type_)
        self.target_ids = target_ids
        self.targets = None

    def resolve_references(self, ann_by_id):
        self.targets = [ann_by_id[id_] for id_ in self.target_ids]


class Note(Annotation):
    def __init__(self, id_, type_, target_id, text):
        Annotation.__init__(self, id_, type_)
        self.target_id = target_id
        self.text = text
        self.target = None

    def resolve_references(self, ann_by_id):
        self.target = ann_by_id[self.target_id]


def parse_xml(fields):
    id_, type_offsets, text, attributes = fields
    type_offsets = type_offsets.split(' ')
    type_, offsets = type_offsets[0], type_offsets[1:]
    return XMLElement(id_, type_, offsets, text, attributes)


def parse_textbound(fields):
    id_, type_offsets, text = fields
    type_offsets = type_offsets.split(' ', 1)
    type_, offsets = type_offsets
    return Textbound(id_, type_, offsets, text)


def parse_relation(fields):
    # allow a variant where the two initial TAB-separated fields are
    # followed by an extra tab
    if len(fields) == 3 and not fields[2]:
        fields = fields[:2]
    id_, type_args = fields
    type_args = type_args.split(' ')
    type_, args = type_args[0], type_args[1:]
    return Relation(id_, type_, args)


def parse_event(fields):
    id_, type_trigger_args = fields
    type_trigger_args = type_trigger_args.split(' ')
    type_trigger, args = type_trigger_args[0], type_trigger_args[1:]
    type_, trigger = type_trigger.split(':')
    return Event(id_, type_, trigger, args)


def parse_attribute(fields):
    id_, type_target_value = fields
    type_target_value = type_target_value.split(' ')
    if len(type_target_value) == 3:
        type_, target, value = type_target_value
    else:
        type_, target = type_target_value
        value = None
    return Attribute(id_, type_, target, value)


def parse_normalization(fields):
    if len(fields) == 3:
        id_, type_target_ref, reftext = fields
    elif len(fields) == 2:    # Allow missing reference text
        id_, type_target_ref = fields
        reftext = ''
    type_, target, ref = type_target_ref.split(' ')
    return Normalization(id_, type_, target, ref, reftext)


def parse_note(fields):
    id_, type_target, text = fields
    type_, target = type_target.split(' ')
    return Note(id_, type_, target, text)


def parse_equiv(fields):
    id_, type_targets = fields
    type_targets = type_targets.split(' ')
    type_, targets = type_targets[0], type_targets[1:]
    return Equiv(id_, type_, targets)


parse_standoff_func = {
    'T': parse_textbound,
    'R': parse_relation,
    'E': parse_event,
    'N': parse_normalization,
    'M': parse_attribute,
    'A': parse_attribute,
    'X': parse_xml,
    '#': parse_note,
    '*': parse_equiv,
}


def parse_standoff_line(l, ln, fn, options):
    try:
        return parse_standoff_func[l[0]](l.split('\t'))
    except Exception:
        if options.ignore_errors:
            error('failed to parse line {} in {}: {}'.format(ln, fn, l))
            return None
        else:
            raise FormatError('error on line {} in {}: {}'.format(ln, fn, l))


def parse_ann_file(fn, options):
    annotations = []
    with open(fn) as f:
        for ln, l in enumerate(f, start=1):
            l = l.rstrip('\n')
            if not l or l.isspace():
                continue
            ann = parse_standoff_line(l, ln, fn, options)
            if ann is not None:
                annotations.append(ann)
    return annotations


def resolve_references(annotations, options):
    ann_by_id = {}
    for a in annotations:
        ann_by_id[a.id_] = a
    for a in annotations:
        a.resolve_references(ann_by_id)


def load_directory_annotations(directory, options):
    files = os.listdir(directory)
    files = [f for f in files if os.path.splitext(f)[1] == '.ann']

    annotations_by_filename = {}
    for fn in files:
        try:
            annotations = parse_ann_file(os.path.join(directory, fn), options)
            resolve_references(annotations, options)
            annotations_by_filename[os.path.splitext(fn)[0]] = annotations
        except Exception as e:
            error('failed to parse {}: {}'.format(fn, e))
            raise
    return annotations_by_filename


def equivalence_sets(anns):
    equivs = []
    for a in anns:
        if isinstance(a, Equiv):
            equivs.append(set(a.target_ids))
    return equivs


def relations(anns):
    rels = []
    for a in anns:
        if isinstance(a, Relation):
            rels.append(a)
    return rels


def is_symmetric(rel):
    return rel.type_ in SYMMETRIC_RELATION_TYPES


def relations_match(gold_rel, pred_rel, options):
    if gold_rel.type_ != pred_rel.type_:
        return False
    if is_symmetric(gold_rel):
        # Symmetric: ignore roles
        if gold_rel.targets() != pred_rel.targets():
            return False
    else:
        # Not symmetric: roles must match
        if gold_rel.roles() != pred_rel.roles():
            return False
        for r in gold_rel.roles():
            if gold_rel.targets_by_role(r) != pred_rel.targets_by_role(r):
                return False
    return True


def unique_relations(fn, dataset, relations):
    unique, seen = [], set()
    for r in relations:
        arg_str = ' '.join(sorted(['{}:{}'.format(r,i) for r, i in r.arg_ids]))
        rel_str = '{} {}'.format(r.type_, arg_str)
        if rel_str in seen:
            print('Note: dropping redundant {} relation from {}: {}'.format(
                dataset, fn, r), file=sys.stderr)
        else:
            unique.append(r)
            seen.add(rel_str)
    return unique


########## end annotation ##########



def prec_rec_F(tp, fp, fn):
    if tp + fp == 0:
        p = 0.0
    else:
        p = tp / (tp + fp)
    if tp + fn == 0:
        r = 0
    else:
        r = tp / (tp + fn)
    if p+r == 0:
        f = 0.0
    else:
        f = 2*p*r/(p+r)
    return p, r, f


def report(tp, fp, fn, label=None):
    p, r, f = prec_rec_F(tp, fp, fn)
    if label is not None:
        print(label, end=' ')
    print('precision {:.1%} ({}/{}) recall {:.1%} ({}/{}) F {:.1%}'.format(
        p, tp, tp+fp, r, tp, tp+fn, f))


def compare_files(fn, gold, pred, options):
    gold_equivs = equivalence_sets(gold)
    pred_equivs = equivalence_sets(pred)
    if pred_equivs and pred_equivs != gold_equivs:
        print('warning: pred Equivs do not match gold, ignoring pred Equivs',
              file=sys.stderr)
    gold_relations = relations(gold)
    pred_relations = relations(pred)

    if options.relations is not None:
        gold_relations = [r for r in gold_relations
                          if r.type_ in options.relations]
        pred_relations = [r for r in pred_relations
                          if r.type_ in options.relations]
    if options.entities is not None:
        gold_relations = [r for r in gold_relations
                          if all(a.type_ in options.entities
                                 for a in r.target_annotations())]
        pred_relations = [r for r in pred_relations
                          if all(a.type_ in options.entities
                                 for a in r.target_annotations())]

    for rels in (gold_relations, pred_relations):
        for rel in rels:
            rel.resolve_equivs(gold_equivs)
    
    gold_relations = unique_relations(fn, 'gold', gold_relations)
    pred_relations = unique_relations(fn, 'pred', pred_relations)
        
    for g in gold_relations:
        for p in pred_relations:
            if relations_match(g, p, options):
                g.matched = True
                p.matched = True
                #print(g, p)

    positive = [r for r in gold_relations if r.matched]
    false_positive = [r for r in pred_relations if not r.matched]
    false_negative = [r for r in gold_relations if not r.matched]

    if options.verbose:
        for label, rels in (('TP', positive),
                            ('FP', false_positive),
                            ('FN', false_negative)):
            for r in rels:
                print('{}:\t{}\t{}'.format(label, fn, r))

    tp, fp, fn = len(positive), len(false_positive), len(false_negative)
    return tp, fp, fn


def compare_datasets(gold, pred, options):
    missing = set(gold.keys()) - set(pred.keys())
    if missing and not options.allow_missing:
        print('error: missing gold files in pred:', ' '.join(missing))
        return 0, 0, 0
    extra = set(pred.keys()) - set(gold.keys())
    if extra:
        print('warning: extra files in pred', ' '.join(extra), file=sys.stderr)

    total_tp, total_fp, total_fn = 0, 0, 0
    for f in gold:
        if f not in pred:
            continue
        tp, fp, fn = compare_files(f, gold[f], pred[f], options)
        if options.verbose:
            report(tp, fp, fn, '{}:'.format(f))
        total_tp += tp
        total_fp += fp
        total_fn += fn
    return total_tp, total_fp, total_fn


def main(argv):
    args = argparser().parse_args(argv[1:])
    if args.relations is not None:
        args.relations = set(args.relations.split(','))
    if args.entities is not None:
        args.entities = set(args.entities.split(','))        
    gold = load_directory_annotations(args.gold, args)
    pred = load_directory_annotations(args.pred, args)
    tp, fp, fn = compare_datasets(gold, pred, args)
    report(tp, fp, fn, 'TOTAL:')


if __name__ == '__main__':
    sys.exit(main(sys.argv))
