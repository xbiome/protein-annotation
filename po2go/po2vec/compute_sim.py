import functools
import multiprocessing as mp
import os

import click as ck
import joblib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from numpy import array, dot, exp, mean, sqrt
from pygosemsim import annotation, download, graph, similarity, term_set
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

G = graph.from_resource('go-basic')
similarity.precalc_lower_bounds(G)

obo_file = 'pygosemsim/_resources/go-basic.obo'

fp = open(obo_file, 'r')
obo_txt = fp.read()
fp.close()
#
obo_txt = obo_txt[obo_txt.find('[Term]') - 1:]
obo_txt = obo_txt[:obo_txt.find('[Typedef]')]
# obo_dict=parse_obo_txt(obo_txt)
id_namespace_dicts = {}
id_name_dicts = {}

for Term_txt in obo_txt.split('[Term]\n'):
    if not Term_txt.strip():
        continue
    name = ''
    ids = []
    for line in Term_txt.splitlines():
        if line.startswith('id: '):
            ids.append(line[len('id: '):])
        elif line.startswith('name: '):
            name = line[len('name: ')]
        elif line.startswith('namespace: '):
            name_space = line[len('namespace: '):]
        elif line.startswith('alt_id: '):
            ids.append(line[len('alt_id: '):])

    for t_id in ids:
        id_namespace_dicts[t_id] = name_space
        id_name_dicts[t_id] = name

# get all used goid
obo_file = 'pygosemsim/_resources/go-basic.obo'

fp = open(obo_file, 'r')
obo_txt = fp.read()
fp.close()
#
obo_txt = obo_txt[obo_txt.find('[Term]') - 1:]
obo_txt = obo_txt[:obo_txt.find('[Typedef]')]

alt_id2id = {}

for Term_txt in obo_txt.split('[Term]\n'):
    if not Term_txt.strip():
        continue
    name = ''
    ids = []
    for line in Term_txt.splitlines():
        if line.startswith('id: '):
            ids.append(line[len('id: '):])

        elif line.startswith('alt_id: '):
            ids.append(line[len('alt_id: '):])
    if len(ids) > 1:
        for t_id in ids[1:]:
            alt_id2id[t_id] = ids[0]


def arcosh(x):
    return np.log(x + (x**2 - 1)**0.5)


def poincareDis(u, v):
    squ = np.linalg.norm(u)
    sqv = np.linalg.norm(v)
    squv = np.linalg.norm(u - v)
    all = 1 + 2 * (squv**2) / ((1 - squ) * (1 - sqv))
    return arcosh(all)


def poin_BMA(sent1, sent2):
    au1 = [min([poincareDis(s, t) for t in sent2]) for s in sent1]
    au2 = [min([poincareDis(s, t) for t in sent1]) for s in sent2]
    data = round((mean(au1) + mean(au2)) / 2.0, 5)
    return -data


def BMA(sent1, sent2):
    au1 = [max([cosine_similarity(s, t)[0][0] for t in sent2]) for s in sent1]
    au2 = [max([cosine_similarity(s, t)[0][0] for t in sent1]) for s in sent2]
    data = round((mean(au1) + mean(au2)) / 2.0, 5)
    return data


def extrtactembedding(dataset_file_path,
                      anno_dict,
                      emb_dict,
                      ont,
                      num_proj_hidden=256):
    measure_df = pd.read_csv(dataset_file_path, delimiter=';')
    all_protein = list(measure_df['Uniprot ID1']) + list(
        measure_df['Uniprot ID2'])
    all_protein = list(set(all_protein))
    prot_emb_dict = {}
    for prot_id in all_protein:
        embedding = []
        if prot_id in anno_dict.keys():
            all_gos = list(anno_dict[prot_id]['annotation'].keys())

            if ont != 'all':
                if ont in [
                        'biological_process', 'molecular_function',
                        'cellular_component'
                ]:
                    all_gos = [
                        x for x in all_gos if id_namespace_dicts[x] == ont
                    ]

            for go in all_gos:
                if go in emb_dict.keys():
                    embedding.append(emb_dict[go].reshape(1, -1))
                elif go in alt_id2id.keys():
                    embedding.append(emb_dict[alt_id2id[go]].reshape(1, -1))

            if embedding == []:
                embedding.append(np.zeros((1, num_proj_hidden)))
            #                 print(prot_id,all_gos)
            prot_emb_dict[prot_id] = embedding
        else:
            embedding.append(np.zeros((1, num_proj_hidden)))
            prot_emb_dict[prot_id] = embedding
    return prot_emb_dict


def bma_score(dataset_file_path,
              anno_dict,
              prot_emb_dict,
              num_proj_hidden=256,
              m_type='other'):
    measure_df = pd.read_csv(dataset_file_path, delimiter=';')
    bert_scores = []
    pool = mp.Pool(32)

    score = poin_BMA if m_type == 'poincare' else BMA
    bert_scores = pool.starmap(
        score,
        [(prot_emb_dict[row['Uniprot ID1']], prot_emb_dict[row['Uniprot ID2']])
         for index, row in measure_df.iterrows()])

    pool.close()
    return bert_scores


@ck.command()
@ck.option(
    '--ont',
    '-O',
    default='all',
    help='{ont, biological_process, molecular_function, cellular_component}')
@ck.option('--dataset_choice', '-dc', default='ppi', help='{ppi, mf}')
@ck.option('--model',
           '-M',
           default='order',
           help='{anc, order, onto, poincare}')
def main(ont, dataset_choice, model):
    if dataset_choice == 'ppi':
        datasets = ['PPI_EC1', 'PPI_DM1', 'PPI_HS1', 'PPI_SC1']
    elif dataset_choice == 'mf':
        datasets = ['MF_EC1', 'MF_DM1', 'MF_HS1', 'MF_SC1']
    else:
        raise ValueError('choice must be from {ppi, mf}')

    # emb_dict = pd.read_pickle('./data/{}_terms_emb.pkl'.format(model))
    emb_dict = pd.read_pickle('./data/PO2Vec/{}_emb.pkl'.format(model))
    num_proj_hidden = list(emb_dict.values())[0].shape[-1]
    for dataset in datasets:
        if 'HS' in dataset:
            annot = annotation.from_resource('goa_human')
            dataset_file_path = 'kgsim-benchmark/DataSets/' + dataset + '.csv'
            print('HS')
        elif 'EC' in dataset:
            annot = annotation.from_resource('ecocyc')
            dataset_file_path = 'kgsim-benchmark/DataSets/' + dataset + '.csv'
            print('EC')
        elif 'SC' in dataset:
            annot = annotation.from_resource('goa_yeast')
            dataset_file_path = 'kgsim-benchmark/DataSets/' + dataset + '.csv'
            print('SC')
        elif 'DM' in dataset:
            annot = annotation.from_resource('goa_fly')
            dataset_file_path = 'kgsim-benchmark/DataSets/' + dataset + '.csv'
            print('DM')

        prot_emb_dict = extrtactembedding(dataset_file_path, annot, emb_dict,
                                          ont, num_proj_hidden)
        bma_gcn_scores = bma_score(dataset_file_path,
                                   annot,
                                   prot_emb_dict,
                                   num_proj_hidden,
                                   m_type=model)

        result_data_path = './data/PO2Vec/' + dataset + '.pkl'
        if not os.path.exists(result_data_path):
            save_result = pd.read_csv(dataset_file_path, delimiter=';')
        else:
            save_result = pd.read_pickle(result_data_path)
        save_result['{}_BMA'.format(model)] = bma_gcn_scores
        save_result.to_pickle('./data/PO2Vec/' + dataset + '.pkl')


if __name__ == '__main__':
    main()
