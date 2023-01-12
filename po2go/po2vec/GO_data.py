import argparse
import pickle
from collections import defaultdict as ddt
from po2go.po2vec.utils import Ontology


parser = argparse.ArgumentParser(description='extract all terms in a go.obo file.',
                                 add_help=False)
parser.add_argument('--go-file',
                    '-gf',
                    default='data/go.obo',
                    type=str,
                    help='go file downloaded from Gene Ontology website')
parser.add_argument('--terms-file',
                    '-tf',
                    default='data/terms_all.pkl',
                    type=str,
                    help='A DataFrame stored all terms')
parser.add_argument('--out-list-file',
                    '-ol',
                    default='data/contra_part_pairs_all.pkl',
                    type=str,
                    help='extracted pair list stored in pkl file')
parser.add_argument('--out-dict-file',
                    '-od',
                    default='data/contrast_pairs.pkl',
                    type=str,
                    help='extracted pair dict stored in pkl file')


def main(go_file, terms_all_file, out_list_file, out_dict_file):
    # INPUT FILES
    go = Ontology(go_file, with_rels=True, include_alt_ids=False)
    with open(terms_all_file, 'rb') as fd:
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
    bp, cc, mf = set(), set(), set()
    for k, v in root_dict.items():
        if k == "GO:0005575" or "GO:0005575" in v:
            cc.add(k)
        elif k == 'GO:0003674' or 'GO:0003674' in v:
            mf.add(k)
        elif k == "GO:0008150" or "GO:0008150" in v:
            bp.add(k)
    contrast_dict['n_cc'] = cc
    contrast_dict['n_bp'] = bp
    contrast_dict['n_mf'] = mf

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
    # save pairs
    pair_list = list()
    for i in range(len(terms)):
        pair_list.append(get_pairs(terms[i], terms_dict, ancestor_dict, root_dict, parents_dict))
        print('{} is ok'.format(i))

    with open(out_list_file, 'wb') as fd:
        pickle.dump(pair_list, fd)

    with open(out_dict_file, 'wb') as fd:
        pickle.dump(contrast_dict, fd)


def get_pairs(terms, terms_dict, ancestor_dict, root_dict, parents_dict):
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


if __name__ == '__main__':
    args = parser.parse_args()
    main(args.go_file, args.terms_file, args.out_list_file, args.out_dict_file)
