import math
import os
import pickle
from collections import Counter
from collections import defaultdict as ddt
from collections import deque

# fake root 'GO:0000000'
term_fake_root = 'GO:0000000'

# three domains of BP, MF and CC
BIOLOGICAL_PROCESS = 'GO:0008150'
MOLECULAR_FUNCTION = 'GO:0003674'
CELLULAR_COMPONENT = 'GO:0005575'
FUNC_DICT = {
    'cc': CELLULAR_COMPONENT,
    'mf': MOLECULAR_FUNCTION,
    'bp': BIOLOGICAL_PROCESS
}

NAMESPACES = {
    'cc': 'cellular_component',
    'mf': 'molecular_function',
    'bp': 'biological_process'
}

data_base_path = u'./data'


### INPUT FILES ###
# Gene Ontology based on .obo File
class Ontology(object):
    def __init__(self,
                 filename='../data/go-basic.obo',
                 with_rels=False,
                 include_alt_ids=True):
        super().__init__()
        self.ont, self.format_version, self.data_version = self.load(
            filename, with_rels, include_alt_ids)
        self.ic = None

    # ------------------------------------
    def load(self, filename, with_rels, include_alt_ids):
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
                        # add all types of relationships revised
                        if it[0] == 'part_of':
                            obj['part_of'].append(it[1])
                        obj['relationship'].append([it[1], it[0]])
                    elif l[0] == 'name':
                        obj['name'] = l[1]
                    elif l[0] == 'is_obsolete' and l[1] == 'true':
                        obj['is_obsolete'] = True
            if obj is not None:
                ont[obj['id']] = obj
        # dealing with alt_ids, why
        for term_id in list(ont.keys()):
            if include_alt_ids:
                for t_id in ont[term_id]['alt_ids']:
                    ont[t_id] = ont[term_id]
            if ont[term_id]['is_obsolete']:
                del ont[term_id]
        # is_a -> children
        for term_id, val in ont.items():
            if 'children' not in val:
                val['children'] = set()
            for p_id in val['is_a'] + val['part_of']:
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

    # revised 'part_of'
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
                for parent_id in (self.ont[t_id]['is_a'] +
                                  self.ont[t_id]['part_of']):
                    if parent_id in self.ont:
                        q.append(parent_id)
        # terms_set.remove(term_id)
        return term_set

    # revised
    def get_parents(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        for parent_id in (self.ont[term_id]['is_a'] +
                          self.ont[term_id]['part_of']):
            if parent_id in self.ont:
                term_set.add(parent_id)
        return term_set

    # get the root terms(only is_a)
    def get_root_ancestors(self, term_id):
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

    def get_roots(self, term_id):
        if term_id not in self.ont:
            return set()
        root_set = set()
        for term in self.get_root_ancestors(term_id):
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

    # all children
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

    # only one layer children
    def get_child_set(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        if term_id not in term_set:
            for ch_id in self.ont[term_id]['children']:
                term_set.add(ch_id)
        return term_set


obo_file = os.path.join(data_base_path, u'go-basic.obo')

data_path_dict = {}

### INPUT FILES ###
data_path_dict['obo'] = obo_file

for k, v in data_path_dict.items():
    print('{:}: {:} [{:5s}]'.format(k, v, str(os.path.exists(v))))

go = Ontology(data_path_dict['obo'], with_rels=True, include_alt_ids=False)

with open('./data/terms_all.pkl', 'rb') as fd:
    terms = pickle.load(fd)
    terms = list(terms['terms'])

terms_set = set(terms)
terms_dict = {v: i for i, v in enumerate(terms)}

# one layer parents, no self
parents_dict = ddt(set)
for i in range(len(terms)):
    parents_dict[terms[i]] = terms_set.intersection(go.get_parents(terms[i]))

# all ancestors, no self
ancestor_dict = ddt(set)
for i in range(len(terms)):
    temp_set = go.get_ancestors(terms[i])
    temp_set.remove(terms[i])
    ancestor_dict[terms[i]] = terms_set.intersection(temp_set)

# get root
root_set = {'GO:0005575', 'GO:0008150', 'GO:0003674'}
root_dict = ddt(set)
for i in range(len(terms)):
    root_dict[terms[i]] = go.get_roots(terms[i])

for k, v in root_dict.items():
    root_dict[k] = list(v)[0]

child_dict = ddt(set)
for i in range(len(terms)):
    child_dict[terms[i]] = terms_set.intersection(go.get_term_set(terms[i]))

child_one_dict = ddt(set)
for i in range(len(terms)):
    child_one_dict[terms[i]] = terms_set.intersection(
        go.get_child_set(terms[i]))

contrast_dict = ddt(set)
n_child_dict = ddt(list)
count = 0
for i in terms:
    temp_anc_set = ancestor_dict[i]
    temp_child_set = go.get_term_set(i)
    temp_list = list()
    for j in go.get_term_set(root_dict[i]):
        if j not in temp_anc_set and j not in temp_child_set:
            temp_list.append(terms_dict[j])
    n_child_dict[terms_dict[i]] = temp_list[:]
    print('{} is ok'.format(count))
    count += 1
contrast_dict = {**contrast_dict, **root_dict, **n_child_dict}


# v3
def get_pairs(terms):
    pair_rank = list()
    temp_list = [terms_dict[terms], -1, []]
    for item in ancestor_dict[terms]:
        if root_dict[item] != root_dict[terms]:
            continue
        third_list = []
        temp_list[1] = terms_dict[item]
        temp = list()
        if parents_dict[item] is not None:
            for j in parents_dict[item]:
                temp.append(terms_dict[j])
        third_list.append(list(temp[:]))
        temp_list[2] = third_list[:]
        pair_rank.append(temp_list[:])
    return pair_rank


# save pairs
pair_list = list()
for i in range(len(terms)):
    pair_list.append(get_pairs(terms[i]))
    print('{} is ok'.format(i))

with open('./data/contra_part_pairs_all.pkl', 'wb') as fd:
    pickle.dump(pair_list, fd)

with open('./data/contrast_pairs.pkl', 'wb') as fd:
    pickle.dump(contrast_dict, fd)
