import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import auc, roc_curve


def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)
    return roc_auc


def calculate_pccs(x, y):
    d1 = (x - np.mean(x)) / np.std(x)
    d2 = (y - np.mean(y)) / np.std(y)
    pccs = np.sum(d1 * d2) / len(x)
    return pccs


sim_from = 'embedding'  # {embedding, sim-method}
dataset_choice = 'ppi'
model = 'order_0.5'  # {anc, order, onto, poincare}
sim_method = 'Resnik'  # {Resnik, Seco}
sim_method_fuse = 'GIC'  # {GIC, BMA}
sim_choice = 'pear'  # {spea, kend, pear, roc}
if dataset_choice == 'ppi':
    datasets = ['PPI_EC1', 'PPI_DM1', 'PPI_HS1', 'PPI_SC1']
elif dataset_choice == 'mf':
    datasets = ['MF_EC1', 'MF_DM1', 'MF_HS1', 'MF_SC1']
else:
    raise ValueError('choice must be from {ppi, mf}')

for dataset in datasets:
    if 'HS' in dataset:
        dataset_file_path = './data/PO2Vec/' + dataset + '.pkl'
        print('HS')
    elif 'EC' in dataset:
        dataset_file_path = './data/PO2Vec/' + dataset + '.pkl'
        print('EC')
    elif 'SC' in dataset:
        dataset_file_path = './data/PO2Vec/' + dataset + '.pkl'
        print('SC')
    elif 'DM' in dataset:
        dataset_file_path = './data/PO2Vec/' + dataset + '.pkl'
        print('DM')

    sim_result = pd.read_pickle(dataset_file_path)
    if dataset_choice == 'mf':
        interaction = np.array(list(sim_result['Sim(Pfam)']))
    else:
        interaction = np.array(list(sim_result['Interaction']))
    if sim_from == 'embedding':
        sim_BMA = np.array(list(sim_result['{}_BMA'.format(model)]))
    else:
        sim_BMA = np.array(
            list(sim_result['{} {}'.format(sim_method_fuse, sim_method)]))
    if sim_choice == 'pear':
        pccs = calculate_pccs(interaction, sim_BMA)
    elif sim_choice == 'spea':
        pccs = stats.spearmanr(interaction, sim_BMA)[0]
    elif sim_choice == 'kend':
        pccs = stats.kendalltau(interaction, sim_BMA)[0]
    elif sim_choice == 'roc':
        pccs = compute_roc(interaction, sim_BMA)
    else:
        raise ValueError('sim_choice is wrong.')
    print(pccs)

interaction_all = []
sim_all = []
for dataset in datasets:
    if 'HS' in dataset:
        dataset_file_path = './data/PO2Vec/' + dataset + '.pkl'
        print('HS')
    elif 'EC' in dataset:
        dataset_file_path = './data/PO2Vec/' + dataset + '.pkl'
        print('EC')
    elif 'SC' in dataset:
        dataset_file_path = './data/PO2Vec/' + dataset + '.pkl'
        print('SC')
    elif 'DM' in dataset:
        dataset_file_path = './data/PO2Vec/' + dataset + '.pkl'
        print('DM')

    sim_result = pd.read_pickle(dataset_file_path)
    if dataset_choice == 'mf':
        interaction_all += list(sim_result['Sim(Pfam)'])
    else:
        interaction_all += list(sim_result['Interaction'])
    if sim_from == 'embedding':
        sim_all += list(sim_result['{}_BMA'.format(model)])
    else:
        sim_all += list(sim_result['{} {}'.format(sim_method_fuse,
                                                  sim_method)])

interaction_np = np.array(interaction_all)
sim_np = np.array(sim_all)
if sim_choice == 'pear':
    pccs = calculate_pccs(interaction_np, sim_np)
elif sim_choice == 'spea':
    pccs = stats.spearmanr(interaction_np, sim_np)[0]
elif sim_choice == 'kend':
    pccs = stats.kendalltau(interaction_np, sim_np)[0]
elif sim_choice == 'roc':
    pccs = compute_roc(interaction_np, sim_np)
else:
    raise ValueError('sim_choice is wrong.')
print(pccs)
