import argparse
from typing import Sequence
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from po2go.utils.esm_dataset import EsmDataset
from po2go.po2go.po2go import PO2GO
from po2go.utils.model import load_model_checkpoint
from po2go.utils.ontology import Ontology

# 输入文件：fasta蛋白序列
# 输出文件：csv表格，基因，功能，置信度

parser = argparse.ArgumentParser(
    description='Protein function Classification config')

parser.add_argument('-i',
                    '--input-file',
                    default='data/protein.fa',
                    type=str,
                    help='A fasta format protein sequence file')
parser.add_argument('-o',
                    '--output-file',
                    default='data/fasta_prdiction.csv',
                    type=str,
                    help='A csv format prediction results')
parser.add_argument('--terms-file',
                    '-tf',
                    default='/home/wangbin/protein-annotation/data/terms_annotated_embeddings.pkl',
                    type=str,
                    metavar='PATH',
                    help='path to predicted terms with corresponding embedding')
parser.add_argument('--resume',
                    default='/home/wangbin/protein-annotation/work_dirs/po2go_annotated/model_best.pth.tar',
                    type=str,
                    metavar='PATH',
                    help='path to best checkpoint (default: none)')
parser.add_argument('-t',
                    '--threshold',
                    default=0.46,
                    type=float,
                    help='Thresholds affecting prediction sensitivity')
parser.add_argument('-b',
                    '--batchsize',
                    default=64,
                    type=int,
                    help='Batchsize size when encoding protein embedding with backbone')


def read_fasta(filename):
    """read fasta file.

    Args:
        filename (str): path of gene sequence file

    Returns:
        tuple: tuple of (gene_names, sequences), gene_names and sequences is list.
    """
    seqs = list()
    names = list()
    seq = ''
    name = ''
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            line = line.rstrip('*')
            if line.startswith('>'):
                if seq != '':
                    seqs.append(seq)
                    names.append(name)
                    seq = ''
                name = line.split('#')[0][1:].rstrip()
            else:
                seq += line
        seqs.append(seq)
        names.append(name)
    return names, seqs


def get_protein_embedding(test_df: pd.DataFrame, batchsize):
    """get embedding for each protein.

    Args:
        test_df (pd.DataFrame): protein sequence DataFrame

    Returns:
        pd.DataFrame: protein embedding DataFrame
    """

    dataset = EsmDataset(test_df)
    loader = DataLoader(dataset,
                        batch_size=batchsize,
                        num_workers=4,
                        collate_fn=dataset.collate_fn)

    backbone = dataset.model
    for p in backbone.parameters():
        p.requires_grad = False
    backbone = backbone.cuda()

    batch_mean_list = []
    for batch in tqdm(loader):
        batch = {key: val.cuda() for key, val in batch.items()}
        batch_aa_embeddings = backbone(batch['input_ids'],
                                       repr_layers=[33])['representations'][33]
        batch_aa_embeddings = batch_aa_embeddings[:, 1:, :]
        masked_lengths = batch['masked_lengths']
        batch_aa_embeddings = batch_aa_embeddings * masked_lengths.unsqueeze(
            -1)
        batch_mean = batch_aa_embeddings.sum(
            dim=1, keepdim=False) / masked_lengths.sum(1).unsqueeze(-1)
        batch_mean = batch_mean.detach().cpu()
        batch_mean_list.append(batch_mean)

        del batch, batch_aa_embeddings, batch_mean
        torch.cuda.empty_cache()

    mean_all = torch.cat(batch_mean_list, dim=0)
    test_df['embeddings'] = list(mean_all.numpy())
    return test_df


class EmbeddingDataset(Dataset):
    """Dataset which used to load protein embedding for prediction."""

    def __init__(self, df):
        super().__init__()
        self.embeddings = list(df['embeddings'])

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        embeddings = torch.from_numpy(np.array(embedding, dtype=np.float32))
        encoded_inputs = {'embeddings': embeddings}
        return encoded_inputs


def get_model(file_terms_annotated: str, model_path, model_args: dict):
    df_terms = pd.read_pickle(file_terms_annotated)
    terms_annotated = df_terms.terms.values
    embeddings = np.concatenate([np.array(embedding, ndmin=2) for embedding in df_terms.embeddings.values])
    terms_embedding = torch.Tensor(embeddings)
    terms_embedding = terms_embedding.cuda()
    model = PO2GO(terms_embedding, **model_args)
    model_state, _ = load_model_checkpoint(model_path)
    model.load_state_dict(model_state)
    model = model.cuda().eval()
    return model, terms_annotated


# 预测

def get_preds(embedding_loader, model):
    pred_all = []
    for batch in embedding_loader:
        batch = {key: val.cuda() for key, val in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            logits = outputs[0]
        preds = logits.sigmoid()
        preds = preds.detach().cpu().numpy()
        pred_all.append(preds)

    pred_all = np.concatenate(pred_all, axis=0)

    return pred_all


# 4.产生预测结果：根据MF、BP、CC的阈值来产生预测结果，合并结果

def preds2go(preds, go, terms: Sequence, th: float):
    filted_terms_annotated = []
    for pred in preds:
        idxs = pred > th
        filted_terms = terms[idxs]
        filted_terms_annotated.append([go_id + '(' + go.get_term(go_id)['name'] + ')' for go_id in filted_terms])
        # filted_terms_annotated.append(filted_terms)
    return filted_terms_annotated


def main(fasta_file, out_file, terms_file, model_path, threshold, batchsize):
    # 1. read fasta, 返回(基因,序列)的dataframe
    # 输入的fasta文件路径
    names, seqs = read_fasta(fasta_file)
    test_df = pd.DataFrame({'names': names, 'sequences': seqs})
    # test_df = pd.read_pickle('/home/wangbin/xbiome-protein-function-prediction/test_dir/uhgg_all_diamond_pred.pkl')
    # 2.生成蛋白质embedding：输入(基因，序列)的dataframe，返回(基因，embedding)的dataframe
    test_df = get_protein_embedding(test_df, batchsize)
    # 3.预测阶段：为每个蛋白质计算terms的概率
    # model_args_annotated = {'protein_dim': 1280, 'latent_dim': 1024, 'prob_predict_temp_dim': 2048}
    model_args_annotated = {'protein_dim': 1280, 'latent_dim': 768, 'prob_predict_temp_dim': 1280}
    model, terms_annotated = get_model(terms_file, model_path, model_args_annotated)
    embedding_dataset = EmbeddingDataset(test_df)
    embedding_loader = DataLoader(embedding_dataset,
                                  batch_size=batchsize,
                                  num_workers=4)

    pred_all = get_preds(embedding_loader, model)
    test_df['annotated_preds'] = list(pred_all)
    terms_path = os.path.dirname(terms_file)
    # 4.产生预测结果：合并结果
    go_file = os.path.join(terms_path, 'go.obo')
    go = Ontology(go_file)
    filted_terms_annotated = preds2go(test_df.annotated_preds, go, terms_annotated, threshold)
    test_df['annotated_preds'] = filted_terms_annotated
    df_pred = test_df[['names', 'annotated_preds']]
    df_pred.to_csv(out_file)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args.input_file, args.output_file, args.terms_file, args.resume, args.threshold, args.batchsize)
