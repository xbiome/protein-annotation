import pandas as pd
import argparse


parser = argparse.ArgumentParser(description='extract annotated with corresponding embedding',
                                 add_help=False)

parser.add_argument('--swissprot-file',
                    '-s',
                    default='data/swissprot.pkl',
                    type=str,
                    help='Path of the file stored processed swissprot protein data.')
parser.add_argument('--embedding-file',
                    '-e',
                    default='data/terms_all_embeddings.pkl',
                    type=str,
                    help='pkl file stored all terms and embeddings, obtained from PO2Vec')
parser.add_argument('--out-file',
                    '-o',
                    default='data/terms_annotated_embeddings.pkl',
                    type=str,
                    help='pkl file stored annotated terms and corresponding embeddings')


def main(protein_file, embedding_file, out_file):
    df_swiss = pd.read_pickle(protein_file)
    annotated_terms = set()
    for _, row in df_swiss.iterrows():
        annotated_terms.update(set(row.prop_annotations))

    terms_annotated = pd.DataFrame({'terms': list(annotated_terms)})
    terms_all_embeddings = pd.read_pickle(embedding_file)
    terms_annotated_embeddings = terms_annotated.merge(terms_all_embeddings, on='terms')
    terms_annotated_embeddings.to_pickle(out_file)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args.swissprot_file, args.embedding_file, args.out_file)
