import math
import random
from collections import Counter, deque

import numpy as np
import torch
from aminoacids import MAXLEN, to_label_index, to_onehot
from numpy.random import randint
from sklearn.metrics import (auc, confusion_matrix, precision_recall_curve,
                             roc_curve)
from torch.utils.data import Dataset

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

# ------------------------------------------------------------------------------------------
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
            if include_alt_id:
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


# Customized pytorch Dateset for annotated sequences
class AnnotatedSequences(Dataset):
    def __init__(self,
                 data_frame,
                 terms,
                 transform=None,
                 target_transform=None,
                 data_type='one-hot'):
        super().__init__()
        # convert terms to dict
        terms_dict = {v: i for i, v in enumerate(terms)}
        # convert to tensor
        if data_type in ['one-hot', 'One-hot']:
            data_tensor, labels = self.df_to_tensor(data_frame, len(terms),
                                                    terms_dict)
        if data_type in ['label-index', 'Label-index']:
            data_tensor, labels = self.df_to_tensor_label_index(
                data_frame, len(terms), terms_dict)
        # self.
        self.data_type = data_type
        self.terms = terms
        self.nb_classes = len(terms)
        self.labels = labels
        self.data = data_tensor
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        if self.data_type in ['one-hot', 'One-hot']:
            return len(self.data)
        if self.data_type in ['label-index', 'Label-index']:
            return self.data.size()[1]

    def __getitem__(self, idx):
        if self.data_type in ['one-hot', 'One-hot']:
            data = self.data[idx]
            label = self.labels[idx]
        if self.data_type in ['label-index', 'Label-index']:
            data = self.data[:, idx]
            label = self.labels[:, idx]
        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            label = self.target_transform(label)
        return data, label

    def df_to_tensor(self, df, nb_classes, terms_dict):
        size = len(df)
        # [batch, num_aa_feature, seq_len]
        data_onehot = np.zeros((len(df), 21, MAXLEN), dtype=np.float32)
        # [batch, num_classes]
        labels = np.zeros((len(df), nb_classes), dtype=np.int32)

        for i, row in enumerate(df.itertuples()):
            seq = row.sequences
            onehot = to_onehot(seq)
            data_onehot[i, :, :] = onehot
            for t_id in row.prop_annotations:
                if t_id in terms_dict:
                    labels[i, terms_dict[t_id]] = 1

        data_onehot = torch.from_numpy(data_onehot).float()
        labels = torch.from_numpy(labels).int()
        return data_onehot, labels

    # data type for transformer
    def df_to_tensor_label_index(self, df, nb_classes, terms_dict):
        size = len(df)
        # [seq_len, batch]
        data_index = np.zeros((MAXLEN, len(df)), dtype=np.float32)
        # [num_classes, batch]
        labels = np.zeros((nb_classes, len(df)), dtype=np.int32)

        for i, row in enumerate(df.itertuples()):
            seq = row.sequences
            label_index = to_label_index(seq)
            data_index[:, i] = label_index
            for t_id in row.prop_annotations:
                if t_id in terms_dict:
                    labels[terms_dict[t_id], i] = 1

        data_index = torch.from_numpy(data_index).int()
        labels = torch.from_numpy(labels).int()
        return data_index, labels


class EsmDataset(Dataset):
    def __init__(self, data_df, terms, batch_converter):
        self.len_df = len(data_df)
        nb_classes = len(terms)
        terms_dict = {v: i for i, v in enumerate(terms)}
        self.data, self.labels = self.data_to_tensor(data_df, terms_dict,
                                                     nb_classes,
                                                     batch_converter)

    def __len__(self):
        return self.len_df

    def __getitem__(self, idx):
        data = self.data[idx, :]
        label = self.labels[idx, :]
        return data, label

    def data_to_tensor(self, data_df, terms_dict, nb_classes, batch_converter):
        labels = np.zeros((len(data_df), nb_classes), dtype=np.int32)
        esm_input = data_df.iloc[:, [1, 3]].values

        convert_labels, convert_strs, convert_tokens = batch_converter(
            esm_input)
        convert_tokens = convert_tokens[:, :1024]

        for i, row in enumerate(data_df.itertuples()):
            for t_id in row.prop_annotations:
                if t_id in terms_dict:
                    labels[i, terms_dict[t_id]] = 1

        labels = torch.from_numpy(labels).int()

        return convert_tokens, labels


class PadEsmDataset(Dataset):
    def __init__(self, data_df, terms, batch_converter):
        self.len_df = len(data_df)
        nb_classes = len(terms)
        terms_dict = {v: i for i, v in enumerate(terms)}
        self.data_df = data_df
        self.data, self.labels = self.data_to_tensor(self.data_df, terms_dict,
                                                     nb_classes,
                                                     batch_converter)

    def __len__(self):
        return self.len_df

    def __getitem__(self, idx):
        data = self.data[idx, :]
        label = self.labels[idx, :]
        seq_mask = torch.from_numpy(np.array(self.data_df.iloc[idx, 3]))
        return data, label, seq_mask

    def data_to_tensor(self, data_df, terms_dict, nb_classes, batch_converter):
        labels = np.zeros((len(data_df), nb_classes), dtype=np.int32)
        esm_input = data_df.iloc[:, [0, 1]].values

        convert_labels, convert_strs, convert_tokens = batch_converter(
            esm_input)
        convert_tokens = convert_tokens[:, :1024]

        for i, row in enumerate(data_df.itertuples()):
            for t_id in row.prop_annotations:
                if t_id in terms_dict:
                    labels[i, terms_dict[t_id]] = 1

        labels = torch.from_numpy(labels).int()

        return convert_tokens, labels


class EsmEmbedDataset(Dataset):
    def __init__(self, data_df):
        self.len_df = len(data_df)
        self.data = data_df

    def __len__(self):
        return self.len_df

    def __getitem__(self, idx):
        data = torch.from_numpy(np.array(self.data.iloc[idx, 0]))
        label = torch.from_numpy(np.array(self.data.iloc[idx, 1])).int()
        return data, label


# ------------------------------------------------------------------------------------------
# functions for evaluation
def get_matrix(labels, preds, threshold=0.3):
    preds = preds.flatten()
    preds[preds >= threshold] = 1
    preds = preds.astype('int8')
    tn, fp, fn, tp = confusion_matrix(labels.flatten(), preds).ravel()
    return tn, fp, fn, tp


def get_level_matrix(labels, preds, level, threshold=0.3):
    preds = preds[..., level]
    preds = preds.flatten()
    preds[preds >= threshold] = 1
    preds = preds.astype('int8')
    labels = labels[..., level]
    tn, fp, fn, tp = confusion_matrix(labels.flatten(), preds).ravel()
    return tn, fp, fn, tp


def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)
    return roc_auc


def compute_aupr_level(labels, preds, level):
    labels = labels[..., level]
    preds = preds[..., level]
    precision, recall, _ = precision_recall_curve(labels.flatten(),
                                                  preds.flatten())
    aupr = auc(recall, precision)
    return aupr


def compute_aupr(labels, preds):
    precision, recall, _ = precision_recall_curve(labels.flatten(),
                                                  preds.flatten())
    aupr = auc(recall, precision)
    return aupr


# set random seed
def set_random_seed(seed=10, deterministic=False, benchmark=False):
    # random.seed(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
    if benchmark:
        torch.backends.cudnn.benchmark = True


#################contrast training###########################
class con_pair_dataset(Dataset):
    def __init__(self,
                 con_pair,
                 contrast_dict,
                 terms,
                 terms_dict,
                 neg_num=80,
                 neg=0.5,
                 neg1_len=0.25):
        super().__init__()
        self.len_df = len(con_pair)
        self.n_cc = list(contrast_dict['n_cc'])
        self.n_bp = list(contrast_dict['n_bp'])
        self.n_mf = list(contrast_dict['n_mf'])
        self.terms = terms
        self.contrast_dict = contrast_dict
        self.terms_dict = terms_dict
        self.neg_num = neg_num
        self.con_pair = con_pair
        self.neg = neg
        self.neg1_len = neg1_len

    def __len__(self):
        return self.len_df

    def __getitem__(self, idx):
        terms_list = [self.con_pair[idx][0], self.con_pair[idx][1]]
        negs1 = set()
        neg1_len = min(len(self.con_pair[idx][2][0]),
                       int(self.neg_num * self.neg1_len))
        if neg1_len > 0:
            negs1 = set(random.sample(self.con_pair[idx][2][0], k=neg1_len))
        negs1 = list(negs1)
        random.shuffle(negs1)

        negs2 = set()
        neg2_len = int((self.neg_num - neg1_len) * self.neg)
        if len(self.contrast_dict[self.con_pair[idx][0]]) <= neg2_len:
            negs2 = negs2 | set(
                random.sample(self.contrast_dict[self.con_pair[idx][0]],
                              k=len(
                                  self.contrast_dict[self.con_pair[idx][0]])))
            negs2 = negs2 | set(
                random.sample(self.contrast_dict[self.con_pair[idx][0]],
                              k=neg2_len -
                              len(self.contrast_dict[self.con_pair[idx][0]])))
        else:
            negs2 = negs2 | set(
                random.sample(self.contrast_dict[self.con_pair[idx][0]],
                              k=neg2_len))
        negs2 = list(negs2)
        random.shuffle(negs2)

        neg_len = neg1_len + neg2_len
        neg_num = self.neg_num - neg_len
        negs3 = set()
        if self.contrast_dict[self.terms[terms_list[0]]] == 'GO:0005575':
            while len(negs3) < neg_num // 3:
                m = randint(0, len(self.n_mf) - 1)
                if self.terms_dict[self.n_mf[m]] not in negs3:
                    negs3.add(self.terms_dict[self.n_mf[m]])
            while len(negs3) < neg_num:
                m = randint(0, len(self.n_bp) - 1)
                if self.terms_dict[self.n_bp[m]] not in negs3:
                    negs3.add(self.terms_dict[self.n_bp[m]])
        elif self.contrast_dict[self.terms[terms_list[0]]] == 'GO:0003674':
            while len(negs3) < neg_num // 5:
                m = randint(0, len(self.n_cc) - 1)
                if self.terms_dict[self.n_cc[m]] not in negs3:
                    negs3.add(self.terms_dict[self.n_cc[m]])
            while len(negs3) < neg_num:
                m = randint(0, len(self.n_bp) - 1)
                if self.terms_dict[self.n_bp[m]] not in negs3:
                    negs3.add(self.terms_dict[self.n_bp[m]])
        elif self.contrast_dict[self.terms[terms_list[0]]] == 'GO:0008150':
            while len(negs3) < neg_num // 3:
                m = randint(0, len(self.n_cc) - 1)
                if self.terms_dict[self.n_cc[m]] not in negs3:
                    negs3.add(self.terms_dict[self.n_cc[m]])
            while len(negs3) < neg_num:
                m = randint(0, len(self.n_mf) - 1)
                if self.terms_dict[self.n_mf[m]] not in negs3:
                    negs3.add(self.terms_dict[self.n_mf[m]])
        negs3 = list(negs3)
        random.shuffle(negs3)

        neg1_num = [neg1_len for i in range(neg1_len)]
        neg2_num = [neg2_len for i in range(neg2_len)]
        neg3_num = [neg_num for i in range(neg_num)]
        neg_num = neg1_num + neg2_num + neg3_num
        neg_num = 1 / np.array(neg_num)
        terms_list = terms_list + negs1 + negs2 + negs3
        return torch.LongTensor(terms_list).view(
            len(terms_list)), torch.from_numpy(neg_num)
