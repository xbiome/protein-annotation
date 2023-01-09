import math
import os
import pickle
from collections import Counter, deque

import numpy as np
import pandas as pd


# Gene Ontology based on .obo File
class Ontology(object):
    def __init__(self,
                 filename='data/go.obo',
                 with_rels=False,
                 include_alt_id=True):
        super().__init__()
        self.ont, self.format_version, self.data_version = self.load(
            filename, with_rels, include_alt_id)
        self.ic = None

    # ------------------------------------
    def load(self, filename, with_rels, include_alt_id):
        ont = dict()
        format_version = []
        data_version = []
        obj = None
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # format version line
                if line.startswith('format-version:'):
                    l = line.split(': ')
                    format_version = l[1]
                # data version line
                if line.startswith('data-version:'):
                    l = line.split(': ')
                    data_version = l[1]
                # item lines
                if line == '[Term]':
                    if obj is not None:
                        ont[obj['id']] = obj
                    obj = dict()
                    # four types of relations to others: is a, part of, has part, or regulates
                    obj['is_a'] = list()
                    obj['part_of'] = list()
                    obj['relationship'] = list()
                    # alternative GO term id
                    obj['alt_ids'] = list()
                    # is_obsolete
                    obj['is_obsolete'] = False
                    continue
                elif line == '[Typedef]':
                    if obj is not None:
                        ont[obj['id']] = obj
                    obj = None
                else:
                    if obj is None:
                        continue
                    l = line.split(': ')
                    if l[0] == 'id':
                        obj['id'] = l[1]
                    elif l[0] == 'alt_id':
                        obj['alt_ids'].append(l[1])
                    elif l[0] == 'namespace':
                        obj['namespace'] = l[1]
                    elif l[0] == 'is_a':
                        obj['is_a'].append(l[1].split(' ! ')[0])
                    elif with_rels and l[0] == 'relationship':
                        it = l[1].split()
                        # add all types of relationships revised. adjustment
                        if it[0] == 'part_of':
                            obj['part_of'].append(it[1])
                        obj['relationship'].append([it[1], it[0]])
                    elif l[0] == 'name':
                        obj['name'] = l[1]
                    elif l[0] == 'is_obsolete' and l[1] == 'true':
                        obj['is_obsolete'] = True
            if obj is not None:
                ont[obj['id']] = obj
        # dealing with alt_ids. adjustment
        for term_id in list(ont.keys()):
            if not include_alt_id:
                for t_id in ont[term_id]['alt_ids']:
                    ont[t_id] = ont[term_id]
            if ont[term_id]['is_obsolete']:
                del ont[term_id]
        # is_a -> children
        for term_id, val in ont.items():
            if 'children' not in val:
                val['children'] = set()
            for p_id in val['is_a']:
                if p_id in ont:
                    if 'children' not in ont[p_id]:
                        ont[p_id]['children'] = set()
                    ont[p_id]['children'].add(term_id)
        return ont, format_version, data_version

    # ------------------------------------
    def has_term(self, term_id):
        return term_id in self.ont

    def get_term(self, term_id):
        if self.has_term(term_id):
            return self.ont[term_id]
        return None

    def calculate_ic(self, annots):
        cnt = Counter()
        for x in annots:
            cnt.update(x)
        self.ic = {}
        for go_id, n in cnt.items():
            parents = self.get_parents(go_id)
            if len(parents) == 0:
                min_n = n
            else:
                min_n = min([cnt[x] for x in parents])

            self.ic[go_id] = math.log(min_n / n, 2)

    def get_ic(self, go_id):
        if self.ic is None:
            raise Exception('Not yet calculated')
        if go_id not in self.ic:
            return 0.0
        return self.ic[go_id]

    def get_ancestors(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        q = deque()
        q.append(term_id)
        while (len(q) > 0):
            t_id = q.popleft()
            if t_id not in term_set:
                term_set.add(t_id)
                for parent_id in self.ont[t_id]['is_a']:
                    if parent_id in self.ont:
                        q.append(parent_id)
        # terms_set.remove(term_id)
        return term_set

    # adjustment
    def get_parents(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        for parent_id in self.ont[term_id]['is_a']:
            if parent_id in self.ont:
                term_set.add(parent_id)
        return term_set

    def get_part_of(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        for parent_id in self.ont[term_id]['part_of']:
            if parent_id in self.ont:
                term_set.add(parent_id)
        return term_set

    # get the root terms
    def get_roots(self, term_id):
        if term_id not in self.ont:
            return set()
        root_set = set()
        for term in self.get_ancestors(term_id):
            if term not in self.ont:
                continue
            if len(self.get_parents(term)) == 0:
                root_set.add(term)

        return root_set

    def get_namespace_terms(self, namespace):
        terms = set()
        for go_id, obj in self.ont.items():
            if obj['namespace'] == namespace:
                terms.add(go_id)
        return terms

    def get_namespace(self, term_id):
        return self.ont[term_id]['namespace']

    def get_term_set(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        q = deque()
        q.append(term_id)
        while len(q) > 0:
            t_id = q.popleft()
            if t_id not in term_set:
                term_set.add(t_id)
                for ch_id in self.ont[t_id]['children']:
                    q.append(ch_id)
        return term_set

    # adjustment
    def get_child_set(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        if term_id not in term_set:
            for ch_id in self.ont[term_id]['children']:
                term_set.add(ch_id)
        return term_set

    # adjustment
    def transmit(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        q = deque()
        q.append(term_id)
        while (len(q) > 0):
            t_id = q.popleft()
            if t_id not in term_set:
                term_set.add(t_id)
                for parent_id in (self.ont[t_id]['is_a'] +
                                  self.ont[t_id]['part_of']):
                    if parent_id in self.ont:
                        q.append(parent_id)
        # terms_set.remove(term_id)
        return term_set
