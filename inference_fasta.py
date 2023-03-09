import argparse
import os
from typing import Sequence
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from po2go.utils.esm_dataset import EsmDataset
from po2go.po2go.po2go import PO2GO
from po2go.utils.model import load_model_checkpoint

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
parser.add_argument('--terms-path',
                    '-tp',
                    default='/home/wangbin/protein-annotation/data/',
                    type=str,
                    metavar='PATH',
                    help='path to predicted terms with corresponding embedding')
parser.add_argument('--model-path',
                    '-mp',
                    default='/home/wangbin/protein-annotation/work_dirs',
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--batchsize',
                    '-b',
                    default=64,
                    type=int,
                    help='Batchsize size when encoding protein embedding with backbone')


def read_fasta(filename):
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


def get_namespace_model(namespace: str, model_args: dict, terms_path, model_path):
    # for namespace in ['mfo', 'bpo', 'cco']:
    terms_file_name = f'terms_{namespace}_embeddings.pkl'
    terms = pd.read_pickle(os.path.join(terms_path, terms_file_name))
    embeddings = np.concatenate([np.array(embedding, ndmin=2) for embedding in terms.embeddings.values])
    terms_embedding = torch.Tensor(embeddings)
    terms_embedding = terms_embedding.cuda()
    # label_map = {term:i for i,term in enumerate(terms.terms.values)}
    model = PO2GO(terms_embedding, **model_args)
    checkpoint_path = os.path.join(model_path, f'po2go_swissprot_{namespace}/model_best.pth.tar')
    model_state, _ = load_model_checkpoint(checkpoint_path)
    model.load_state_dict(model_state)
    model = model.cuda().eval()
    return model, terms.terms.values

# 预测


def get_preds(embedding_loader, model_mfo, model_bpo, model_cco):
    pred_all_mfo = []
    pred_all_bpo = []
    pred_all_cco = []
    for batch in embedding_loader:
        batch = {key: val.cuda() for key, val in batch.items()}
        with torch.no_grad():
            outputs = model_mfo(**batch)
            logits = outputs[0]
        preds = logits.sigmoid()
        preds = preds.detach().cpu().numpy()
        pred_all_mfo.append(preds)
        with torch.no_grad():
            outputs = model_bpo(**batch)
            logits = outputs[0]
        preds = logits.sigmoid()
        preds = preds.detach().cpu().numpy()
        pred_all_bpo.append(preds)
        with torch.no_grad():
            outputs = model_cco(**batch)
            logits = outputs[0]
        preds = logits.sigmoid()
        preds = preds.detach().cpu().numpy()
        pred_all_cco.append(preds)
    pred_all_mfo = np.concatenate(pred_all_mfo, axis=0)
    pred_all_bpo = np.concatenate(pred_all_bpo, axis=0)
    pred_all_cco = np.concatenate(pred_all_cco, axis=0)

    return pred_all_mfo, pred_all_bpo, pred_all_cco

# 4.产生预测结果：根据MF、BP、CC的阈值来产生预测结果，合并结果


def preds2go(preds, terms: Sequence, th: float):
    filted_terms_annotated = []
    for pred in preds:
        idxs = pred > th
        filted_terms = terms[idxs]
        # go_file = "/home/wangbin/protein-annotation/data/go.obo"
        # go = Ontology(go_file)
        # filted_terms_annotated.append([go_id+'('+go.get_term(go_id)['name']+')' for go_id in filted_terms])
        filted_terms_annotated.append(filted_terms)
    return filted_terms_annotated


def main(fasta_file, out_file, terms_path, model_path, batchsize):
    # 1. read fasta, 返回(基因,序列)的dataframe
    # 输入的fasta文件路径
    names, seqs = read_fasta(fasta_file)
    test_df = pd.DataFrame({'names': names, 'sequences': seqs})
    # test_df = pd.read_pickle('/home/wangbin/xbiome-protein-function-prediction/test_dir/uhgg_all_diamond_pred.pkl')
    # 2.生成蛋白质embedding：输入(基因，序列)的dataframe，返回(基因，embedding)的dataframe
    test_df = get_protein_embedding(test_df, batchsize)
    # 3.预测阶段：为每个蛋白质计算terms的概率
    model_args_mfo = {'protein_dim': 1280, 'latent_dim': 512, 'prob_predict_temp_dim': 768}
    model_args_bpo = {'protein_dim': 1280, 'latent_dim': 768, 'prob_predict_temp_dim': 1280}
    model_args_cco = {'protein_dim': 1280, 'latent_dim': 512, 'prob_predict_temp_dim': 896}
    model_mfo, model_bpo, model_cco = None, None, None
    for namespace in ['mfo', 'bpo', 'cco']:
        exec(f'model_args = model_args_{namespace}')
        exec(f'model_{namespace}, terms_{namespace} = get_namespace_model(namespace, model_args, terms_path, model_path)')

    model_mfo, terms_mfo = get_namespace_model('mfo', model_args_mfo, terms_path, model_path)
    model_bpo, terms_bpo = get_namespace_model('bpo', model_args_bpo, terms_path, model_path)
    model_cco, terms_cco = get_namespace_model('cco', model_args_cco, terms_path, model_path)
    # model, terms_annotated = get_model(terms_file, model_path, model_args_annotated)
    embedding_dataset = EmbeddingDataset(test_df)
    embedding_loader = DataLoader(embedding_dataset,
                                  batch_size=batchsize,
                                  num_workers=4)

    pred_all_mfo, pred_all_bpo, pred_all_cco = get_preds(embedding_loader, model_mfo, model_bpo, model_cco)
    # 4.产生预测结果：合并结果
    th_mfo = 0.52
    th_bpo = 0.444
    th_cco = 0.445
    filted_terms_mfo = preds2go(pred_all_mfo, terms_mfo, th_mfo)
    filted_terms_bpo = preds2go(pred_all_bpo, terms_bpo, th_bpo)
    filted_terms_cco = preds2go(pred_all_cco, terms_cco, th_cco)

    all = []
    for mf, bp, cc in zip(filted_terms_mfo, filted_terms_bpo, filted_terms_cco):
        mf = list(mf)
        bp = list(bp)
        cc = list(cc)
        mf.extend(bp)
        mf.extend(cc)
        all.append(mf)

    test_df['annotated_preds'] = all
    df_pred = test_df[['names', 'annotated_preds']]
    df_pred.to_csv(out_file)


if __name__ == '__main__':
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    args = parser.parse_args()
    main(args.input_file, args.output_file, args.terms_path, args.model_path, args.batchsize)
