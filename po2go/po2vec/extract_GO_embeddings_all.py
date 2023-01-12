import torch
import pickle
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Extract GO embeddings for all terms',
                                 add_help=False)

parser.add_argument('--terms-file',
                    '-tf',
                    default='data/terms_all.pkl',
                    type=str,
                    help='A DataFrame stored all terms')
parser.add_argument('--model-file',
                    '-mf',
                    default='models/part_order_400.pth',
                    type=str,
                    help='model weights file stored embedidngs of all terms')
parser.add_argument('--out-file',
                    '-of',
                    default='data/terms_all_embeddings.pkl',
                    type=str,
                    help='A DataFrame stored all embeddings corresponding to terms')


def main(terms_file, model_file, out_file):
    model = torch.load(model_file, map_location="cpu")
    proj_con = model["embedding"].numpy()

    # The file contains all terms you need
    with open(terms_file, "rb") as fd:
        terms_all = pickle.load(fd)
        terms_all = list(terms_all["terms"])

    terms_emb_dict = {}
    for i in range(len(terms_all)):
        terms_emb_dict[terms_all[i]] = proj_con[i]

    df_terms_all_embeddings = pd.DataFrame({'terms': terms_emb_dict.keys(), 'embeddings': terms_emb_dict.values()})
    df_terms_all_embeddings.to_pickle(out_file)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args.terms_file, args.model_file, args.out_file)
