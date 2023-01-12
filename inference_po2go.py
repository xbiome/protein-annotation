import argparse
import logging
import os
import sys
import time

import pandas as pd
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import yaml
from torch.utils.data import DataLoader
from po2go.po2go.po2go import PO2GO

from po2go.utils.embedding_dataset import EmbeddingDataset
from po2go.po2go.trainer.training import predict
from po2go.utils.model import load_model_checkpoint
from collections import OrderedDict


sys.path.append('../')

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config',
                                                 add_help=False)
parser.add_argument('-c',
                    '--config',
                    default='',
                    type=str,
                    metavar='FILE',
                    help='YAML config file specifying default arguments')
parser = argparse.ArgumentParser(
    description='Protein function Classification Model Train config')
parser.add_argument('--data-path',
                    default='data',
                    type=str,
                    help='data dir of dataset')
parser.add_argument('--model',
                    metavar='MODEL',
                    default='po2go',
                    help='model architecture: (default: po2go)')
parser.add_argument('--resume',
                    default=None,
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-ld',
                    '--latent-dim',
                    default=768,
                    type=int,
                    metavar='LD',
                    help='latent-dim')

parser.add_argument('-p',
                    '--predict-temp-dim',
                    default=1280,
                    type=int,
                    metavar='PTD',
                    help='prob-predict-temp-dim')
parser.add_argument('-e',
                    '--evaluation',
                    default=True,
                    type=bool,
                    help='Do evaluation after inference')
parser.add_argument('--namespace',
                    default='mfo',
                    type=str,
                    help='cco, mfo, bpo, annotated')

parser.add_argument(
    '--amp',
    action='store_true',
    default=False,
    help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('-j',
                    '--workers',
                    type=int,
                    default=4,
                    metavar='N',
                    help='how many training processes to use (default: 1)')
parser.add_argument('-b',
                    '--batch-size',
                    default=256,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 256) per gpu')
parser.add_argument('--output-dir',
                    default='./work_dirs',
                    type=str,
                    help='output directory for model and log')


def main(args):
    args.gpu = 0
    terms_file_name = f'terms_{args.namespace}_embeddings.pkl'
    terms = pd.read_pickle(os.path.join(args.data_path, terms_file_name))
    embeddings = np.concatenate([
        np.array(embedding, ndmin=2) for embedding in terms.embeddings.values
    ])
    terms_embedding = torch.Tensor(embeddings)
    terms_embedding = terms_embedding.cuda()
    label_map = {term: i for i, term in enumerate(terms.terms.values)}

    # Dataset and DataLoader
    try:
        test_dataset = EmbeddingDataset(
            label_map,
            root_path=args.data_path,
            file_name=f'{args.namespace}_test_embeddings.pkl')
    except FileNotFoundError:
        test_dataset = EmbeddingDataset(
            label_map,
            root_path=args.data_path,
            file_name='test_data_embedding.pkl')
    # dataloders
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.workers,
                             pin_memory=True)

    model = PO2GO(terms_emb=terms_embedding,
                  protein_dim=1280,
                  latent_dim=args.latent_dim,
                  prob_predict_temp_dim=args.predict_temp_dim)

    if args.resume is not None:
        if args.local_rank == 0:
            model_state, optimizer_state = load_model_checkpoint(args.resume)
            model.load_state_dict(model_state)

    # define loss function (criterion) and optimizer
    model = model.cuda()
    if args.evaluation:
        # run predict
        predictions, test_metrics = predict(model,
                                            test_loader,
                                            use_amp=args.amp,
                                            logger=logger)
        logger.info('Test metrics: %s' % (test_metrics))
        test_data_path = os.path.join(args.data_path,
                                      f'{args.namespace}_test_embeddings.pkl')
        try:
            test_df = pd.read_pickle(test_data_path)
        except FileNotFoundError:
            test_df = pd.read_pickle(os.path.join(args.data_path, 'test_data_embedding.pkl'))

        preds, test_labels = predictions
        test_df['labels'] = list(test_labels)
        test_df['preds'] = list(preds)
        df_path = os.path.join(args.data_path,
                               args.namespace + '_p2go_predictions' + '.pkl')
        test_df.to_pickle(df_path)
        logger.info(f'Saving results to {df_path}')
        true_labels = np.concatenate(test_labels, axis=0)
        pred_labels = np.concatenate(preds, axis=0)
        # avg_auc
        max_f1, aupr, max_f1_thresh = compute_aupr_fmax(true_labels, pred_labels)
        metrics = OrderedDict([('fmax', max_f1), ('aupr', aupr), ('threshold', max_f1_thresh)])
        logger.info('Test metrics: %s' % (metrics))
    else:
        "not implement."


from sklearn import metrics


def compute_aupr_fmax(labels, preds):
    precision, recall, threshold = metrics.precision_recall_curve(labels.flatten(), preds.flatten())
    numerator = 2 * recall * precision
    denom = recall + precision
    f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom != 0))
    max_f1 = np.max(f1_scores)
    max_f1_thresh = threshold[np.argmax(f1_scores)]
    aupr = metrics.auc(recall, precision)
    return max_f1, aupr, max_f1_thresh


def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


if __name__ == '__main__':
    args, args_text = _parse_args()
    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

    task_name = args.model + '_' + args.namespace
    args.output_dir = os.path.join(args.output_dir, task_name)
    if not torch.distributed.is_initialized() or torch.distributed.get_rank(
    ) == 0:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    with open(os.path.join(args.output_dir, 'args.yaml'), 'w') as f:
        f.write(args_text)

    logger = logging.getLogger('')
    filehandler = logging.FileHandler(
        os.path.join(args.output_dir, 'summary.log'))
    streamhandler = logging.StreamHandler()
    logger.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)
    cudnn.benchmark = True
    start_time = time.time()
    main(args)
