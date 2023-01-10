import torch
import argparse
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from po2go.utils.esm_dataset import EsmDataset

parser = argparse.ArgumentParser(description='get esm embedding from pickle file',
                                 add_help=False)

parser.add_argument('--input-file',
                    '-i',
                    default='data/swissprot.pkl',
                    type=str,
                    help='Path of the file store protein sequences with pkl format')
parser.add_argument('--out-file',
                    '-o',
                    default='data/swissprot_embedding.pkl',
                    type=str,
                    help='Result in pkl format, which add a column stored esm embedding for each protein')
parser.add_argument('--batch-size',
                    '-b',
                    default=32,
                    type=int,
                    help='batch size used to generate embedding by esm-1b')


def main(input_file, out_file, batchsize=32):
    df = pd.read_pickle(input_file)
    df = get_protein_embedding(df, batchsize=batchsize)
    df.to_pickle(out_file)


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


if __name__ == '__main__':
    args = parser.parse_args()
    main(args.input_file, args.out_file, args.batch_size)
