import argparse
import logging
import math
import os
import pickle
import random
from contextlib import contextmanager

import esm
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from losses import ConPairLoss
from utils import con_pair_dataset, set_random_seed

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'


# 0.6513
def main():
    parser = argparse.ArgumentParser(description='weight sample pianxu')
    parser.add_argument('--data_path',
                        '-dp',
                        default='data',
                        help='Path to store data')
    parser.add_argument('--model_path',
                        '-mp',
                        default='models/weight_triple',
                        help='Path to save model')
    parser.add_argument('--summary_path',
                        '-sp',
                        default='logs/triple',
                        help='Path to save summary')
    parser.add_argument('--model_load',
                        '-ml',
                        type=int,
                        default=0,
                        help='Load model epoch and the model must exist')
    parser.add_argument('--local_rank', type=int, default=-1)

    args = parser.parse_args()

    # check data_path
    if not os.path.exists(args.data_path):
        print('Unable to find data path %s.' % args.data_path)

    # check model_path
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
        print('Create %s to save model.' % args.model_path)
    # filename format to save or load model
    model_prefix = r'part_order_'
    model_suffix = r'.pth'
    model_file = os.path.join(args.model_path,
                              model_prefix + r'%d' + model_suffix)

    # check summary_path
    if not os.path.exists(args.summary_path):
        os.makedirs(args.summary_path)
        print('Create %s to save summary.' % args.summary_path)

    # log setting
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    f_handler = logging.FileHandler(
        os.path.join(args.summary_path, 'exper_triple.txt'))
    logger.addHandler(f_handler)

    ### INPUT FILES ###
    # Gene Ontology file in OBO Format
    go_file = os.path.join(args.data_path, u'go-basic.obo')
    # Result file with a list of terms for prediction task
    out_terms_file = os.path.join(args.data_path, u'terms_all.pkl')
    pair_file = os.path.join(args.data_path, u'contra_part_pairs_all.pkl')
    contrast_file = os.path.join(args.data_path, u'contrast_pairs.pkl')

    data_path_dict = {}

    ### INPUT FILES ###
    data_path_dict['go'] = go_file
    data_path_dict['terms'] = out_terms_file

    # Hyper parameters of model and training
    params = {
        'learning_rate': 5e-2,  # 学习率
        'epochs': 400,  # 迭代次数
        'train_batch_size': 3000,  # 训练集一个batch大小
    }

    args.device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu')
    # set random seed
    set_random_seed(42)

    # Terms of annotations
    terms_df = pd.read_pickle(data_path_dict['terms'])
    terms = terms_df['terms'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}
    nb_classes = len(terms)

    with open(pair_file, 'rb') as fd:
        pair_list_file = pickle.load(fd)

    pair_list = []
    for i in range(len(pair_list_file)):
        pair_list += pair_list_file[i]

    with open(contrast_file, 'rb') as fd:
        contrast_dict = pickle.load(fd)

    neg_num = 80
    pair_data = con_pair_dataset(pair_list,
                                 contrast_dict,
                                 terms,
                                 terms_dict,
                                 neg_num=neg_num,
                                 neg=0.5,
                                 neg1_len=0.25)
    print('Size of train dataset:', len(pair_data))

    train_dataloader = DataLoader(pair_data,
                                  batch_size=params['train_batch_size'],
                                  shuffle=True)

    # model
    model = PairModel(nb_classes, neg_num + 2)
    model.to(args.device)

    # check the previous saved models
    files = os.listdir(args.model_path)
    epoch_list = [
        int(f[len(model_prefix):-len(model_suffix)]) for f in files
        if f[:len(model_prefix)] == model_prefix
    ]
    if len(epoch_list) > 0 and args.model_load > 0:
        max_epoch_file = model_file % args.model_load if args.model_load in epoch_list else ''
    else:
        max_epoch_file = r''

    # load the last saved model to continue training
    checkpoint = None
    if os.path.exists(max_epoch_file):
        checkpoint = torch.load(max_epoch_file, map_location=args.device)
        model.load_state_dict(checkpoint['net'], strict=True)
        print('Load model from file:', max_epoch_file)
        start_epoch = args.model_load
    else:
        print('No model to load.')
        start_epoch = 0

    crition = ConPairLoss(neg_num=neg_num)

    # Adam
    optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                  model.parameters()),
                           lr=params['learning_rate'])
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    print('Training starts:')
    for epoch in range(start_epoch + 1, params['epochs'] + 1):
        print('--------Epoch %02d--------' % epoch)
        logger.info('--------Epoch %02d--------\n' % epoch)
        train_loss = train(model, args.device, optimizer, crition,
                           train_dataloader, args)

        logger.info('train_loss:{}, epoch:{}\n'.format(train_loss, epoch))

        # save model
        if epoch % 40 == 0:
            checkpoint = {
                'net': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'embedding': model.embedding.weight.data
            }
            torch.save(checkpoint, model_file % epoch)
            print('Model parameters are saved!')


class PairModel(nn.Module):
    def __init__(self, input_emb, sample_size):
        super().__init__()
        self.emb_dim = 256
        self.embedding = nn.Embedding(input_emb, self.emb_dim)
        self.temp_dim = 1024
        self.final_dim = 512
        self.fc = nn.Linear(self.emb_dim, self.temp_dim)
        self.fc2 = nn.Linear(self.temp_dim, self.final_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.fc(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = x / torch.norm(x, dim=-1, keepdim=True)
        return x


# Training function
def train(model, device, optimizer, crition, train_dataloader, args):
    model.train()
    train_loss = 0

    for index, (x, rate) in enumerate(tqdm(train_dataloader)):
        x = Variable(x.squeeze().to(device))
        rate = rate.to(device)

        optimizer.zero_grad()
        output = model(x)
        loss = crition(output, rate)
        train_loss += loss.item() * x.shape[0]
        loss.backward()
        optimizer.step()

    train_loss /= len(train_dataloader.dataset)
    print('Train set: Average loss: {:.4f}'.format(train_loss))

    return train_loss


if __name__ == '__main__':
    main()
