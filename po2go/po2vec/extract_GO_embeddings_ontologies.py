import numpy as np
import pandas as pd
import torch
import pickle
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
parser.add_argument('--ontology-term-path',
                    '-otp',
                    default='data',
                    type=str,
                    help='A DataFrame stored terms of a certain ontology, named like "mfo_terms.pkl", "bpo_terms.pkl", "cco_terms.pkl"')
parser.add_argument('--out-embedding-path',
                    '-oep',
                    default='data',
                    type=str,
                    help='A DataFrame stored all embeddings corresponding to terms, \
                        named like "mfo_terms_embeddings.pkl", "bpo_terms_embeddings.pkl", "cco_terms_embeddings.pkl"')


def main(model_file, terms_all_file, ontology_term_path, out_path):
    model = torch.load(model_file, map_location="cpu")
    proj_con = model["embedding"]

    # The file contains all terms you need
    with open(terms_all_file, "rb") as fd:
        terms_all = pickle.load(fd)
        terms_all = list(terms_all["terms"])

    terms_emb_dict = {}
    for i in range(len(terms_all)):
        terms_emb_dict[terms_all[i]] = proj_con[i]

    for ont in {'cc', 'mf', 'bp'}:
        terms_df = pd.read_pickle(f"{ontology_term_path}/{ont}o_terms.pkl")
        terms_emb = np.zeros((len(terms_df), 256), dtype=np.float32)
        # Terms of annotations
        ont_terms = list(terms_df['terms'])
        for i in range(len(ont_terms)):
            terms_emb[i] = terms_emb_dict[ont_terms[i]]

        df_terms_embeddings = pd.DataFrame({'terms': ont_terms, 'embeddings': list(terms_emb)})
        df_terms_embeddings.to_pickle(f"{out_path}/{ont}o_terms_embeddings.pkl")


if __name__ == '__main__':
    args = parser.parse_args()
    main(args.model_file, args.terms_file, args.ontology_term_path, args.out_embedding_path)
