#!/usr/bin/env python

import argparse
import logging
import os
import sys
import numpy as np
import pandas as pd
from sklearn import metrics
from tqdm import tqdm

from partialgo.data.utils.data_utils import FUNC_DICT, NAMESPACES
from partialgo.data.utils.ontology import Ontology


sys.path.append('../')

parser = argparse.ArgumentParser(
    description='Protein function Classification Model Train config')
parser.add_argument('--data-path',
                    '-dp',
                    default='data',
                    type=str,
                    help='data dir of dataset')
parser.add_argument('--train-data-file',
                    '-tf',
                    default='data/mfo_train_data_embeddings.pkl',
                    help='Data file with training features')
parser.add_argument('--pred-file',
                    '-pf',
                    default='data/mfo_po2go_predictions.pkl',
                    help='Data file with test')
parser.add_argument('--go-file',
                    '-gf',
                    default='data/go.obo',
                    help='Ontology file')
parser.add_argument('--output-dir', '-o', default='./evaluation_summary', help='output dir')
parser.add_argument('--ont', default='mf', help='bp, mf, cc')


def main(data_path, train_data_file,
         test_data_file,
         go_obo_file,
         ont, logger):
    print(f'Evaluate {test_data_file}')
    test_df = pd.read_pickle(test_data_file)
    # compute Fmax and AUPR
    labels = test_df.labels.values.flatten()
    preds = test_df.preds.values.flatten()
    all_labels = []
    for item in labels:
        all_labels.append(item)
    all_preds = []
    for item in preds:
        all_preds.append(item)
    precisions, recalls, thresholds = metrics.precision_recall_curve(np.array(all_labels).flatten(), np.array(all_preds).flatten())
    aupr = metrics.auc(recalls, precisions)
    # compute Fmax
    numerator = 2 * recalls * precisions
    denom = recalls + precisions
    f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom != 0))
    max_f1 = np.max(f1_scores)
    max_f1_thresh = thresholds[np.argmax(f1_scores)]
    logger.info(f'Fmax:{max_f1:.3f}    AUPR:{aupr:.3f}    Th:{max_f1_thresh:.3f}')
    # compute Smin
    go_rels = Ontology(go_obo_file, with_rels=True)

    term_path = data_path + f'/terms_{ont}o_embeddings.pkl'
    terms = pd.read_pickle(term_path).terms.values
    train_df = pd.read_pickle(train_data_file)
    annotations = train_df['prop_annotations'].values
    annotations = list(map(lambda x: set(x), annotations))

    test_annotations = test_df['prop_annotations'].values
    test_annotations = list(map(lambda x: set(x), test_annotations))
    go_rels.calculate_ic(annotations + test_annotations)

    ics = {}
    for term in terms:
        ics[term] = go_rels.get_ic(term)
    prot_index = {}
    for i, row in enumerate(train_df.itertuples()):
        prot_index[row.proteins] = i
    model_preds = list(test_df.preds)
    logger.info(f'Evaluate the {ont} protein family')
    smin = evaluate_model_prediction(
        test_annotations, terms, model_preds, go_rels, ont)
    logger.info(f'Fmax:{max_f1:.3f}     Smin:{smin:.3f}    AUPR:{aupr:.3f}     Threshold:{max_f1_thresh:.3f}')


def evaluate_s(go, real_annots, pred_annots):
    total = 0
    ru = 0.0
    mi = 0.0
    for i in range(len(real_annots)):
        if len(real_annots[i]) == 0:
            continue
        tp = set(real_annots[i]).intersection(set(pred_annots[i]))
        fp = pred_annots[i] - tp
        fn = real_annots[i] - tp
        for go_id in fp:
            mi += go.get_ic(go_id)
        for go_id in fn:
            ru += go.get_ic(go_id)
        total += 1
    ru /= total
    mi /= total
    # s = math.sqrt(ru * ru + mi * mi)
    return ru, mi


def evaluate_model_prediction(labels, terms, model_preds, go_rels, ont,
                              ):
    ru_list = []
    mi_list = list()
    # go set
    go_set = go_rels.get_namespace_terms(NAMESPACES[ont])
    go_set.remove(FUNC_DICT[ont])
    # labels
    labels = list(map(lambda x: set(filter(lambda y: y in go_set and y in terms, x)), labels))
    for t in tqdm(range(0, 101, 1)):
        threshold = t / 100.0
        preds = []
        for i, _ in enumerate(model_preds):
            annots = set()
            pred_score = model_preds[i]
            pred_label = terms[pred_score > threshold]
            annots = set(pred_label)
            new_annots = set()
            for go_id in annots:
                new_annots |= go_rels.get_ancestors(go_id)
            preds.append(new_annots)
        # Filter classes
        preds = list(
            map(lambda x: set(filter(lambda y: y in go_set and y in terms, x)), preds))
        ru, mi = evaluate_s(go_rels, labels, preds)
        ru_list.append(ru)
        mi_list.append(mi)
    ru = np.array(ru_list)
    mi = np.array(mi_list)
    smin = np.min(np.sqrt(ru * ru + mi * mi))
    return smin


if __name__ == '__main__':
    args = parser.parse_args()
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    filehandler = logging.FileHandler(
        os.path.join(args.output_dir, 'summary.log'))
    streamhandler = logging.StreamHandler()
    logger.addHandler(streamhandler)
    logger.addHandler(filehandler)
    main(args.data_path, args.train_data_file, args.pred_file, args.go_file, args.ont, logger)
