import argparse
import os
import pandas as pd
from po2go.utils.ontology import Ontology


parser = argparse.ArgumentParser(description='seperate protein data into three groups based ontology',
                                 add_help=False)

parser.add_argument('--train-file',
                    '-train',
                    default='data/train_data_embedding.pkl',
                    type=str,
                    help='input train data file in pkl format')
parser.add_argument('--test-file',
                    '-test',
                    default='data/test_data_embedding.pkl',
                    type=str,
                    help='input test data file in pkl format')
parser.add_argument('--terms-file',
                    '-terms',
                    default='data/terms_all_embeddings.pkl',
                    type=str,
                    help='All terms with PO2Vec embedding in pkl format')
parser.add_argument('--go-file',
                    '-go',
                    default='data/go.obo',
                    type=str,
                    help='All terms with PO2Vec embedding in pkl format')
parser.add_argument('--out-path',
                    '-op',
                    default='data',
                    type=str,
                    help='A folder stored output seperated train and test data files')


def main(train_data_file, test_data_file, terms_embedding_file, go_file, data_path):

    bpo_df, mfo_df, cco_df = seperate(train_data_file, go_file)
    # train data
    bpo_df.to_pickle(os.path.join(data_path, 'bpo_train_embeddings.pkl'))
    mfo_df.to_pickle(os.path.join(data_path, 'mfo_train_embeddings.pkl'))
    cco_df.to_pickle(os.path.join(data_path, 'cco_train_embeddings.pkl'))
    # terms
    terms_all = pd.read_pickle(terms_embedding_file)
    for i, df in enumerate([bpo_df, mfo_df, cco_df]):
        terms_set = set()
        for prop_terms in df.prop_annotations.values:
            terms_set.update(set(prop_terms))
        terms_df = pd.DataFrame({'terms': list(terms_set)})
        terms_embedding = terms_df.merge(terms_all, on='terms')
        if i == 0:
            terms_embedding.to_pickle(os.path.join(data_path, 'terms_bpo_embeddings.pkl'))
        if i == 1:
            terms_embedding.to_pickle(os.path.join(data_path, 'terms_mfo_embeddings.pkl'))
        if i == 2:
            terms_embedding.to_pickle(os.path.join(data_path, 'terms_cco_embeddings.pkl'))
    # test data
    bpo_df, mfo_df, cco_df = seperate(test_data_file, go_file)
    bpo_df.to_pickle(os.path.join(data_path, 'bpo_test_embeddings.pkl'))
    mfo_df.to_pickle(os.path.join(data_path, 'mfo_test_embeddings.pkl'))
    cco_df.to_pickle(os.path.join(data_path, 'cco_test_embeddings.pkl'))


def seperate(data_file, go_file):
    df = pd.read_pickle(data_file)
    ont = Ontology(go_file, with_rels=True, include_alt_id=True)
    bpo_proteins = []
    bpo_sequences = []
    bpo_annotations = []
    bpo_embeddings = []

    mfo_proteins = []
    mfo_sequences = []
    mfo_annotations = []
    mfo_embeddings = []

    cco_proteins = []
    cco_sequences = []
    cco_annotations = []
    cco_embeddings = []

    for item in df.iterrows():
        protien = item[1]['proteins']
        seq = item[1]['sequences']
        annotation = item[1]['prop_annotations']
        embedding = item[1]['embeddings']
        bpo_annotation = []
        mfo_annotation = []
        cco_annotation = []
        for term in annotation:
            if ont.get_namespace(term) == 'biological_process':
                bpo_annotation.append(term)
            elif ont.get_namespace(term) == 'molecular_function':
                mfo_annotation.append(term)
            elif ont.get_namespace(term) == 'cellular_component':
                cco_annotation.append(term)
        if len(bpo_annotation) > 0:
            bpo_proteins.append(protien)
            bpo_sequences.append(seq)
            bpo_annotations.append(bpo_annotation)
            bpo_embeddings.append(embedding)
        if len(mfo_annotation) > 0:
            mfo_proteins.append(protien)
            mfo_sequences.append(seq)
            mfo_annotations.append(mfo_annotation)
            mfo_embeddings.append(embedding)
        if len(cco_annotation) > 0:
            cco_proteins.append(protien)
            cco_sequences.append(seq)
            cco_annotations.append(cco_annotation)
            cco_embeddings.append(embedding)

    bpo_df = pd.DataFrame({
        'proteins': bpo_proteins,
        'sequences': bpo_sequences,
        'prop_annotations': bpo_annotations,
        'embeddings': bpo_embeddings
    })
    mfo_df = pd.DataFrame({
        'proteins': mfo_proteins,
        'sequences': mfo_sequences,
        'prop_annotations': mfo_annotations,
        'embeddings': mfo_embeddings
    })
    cco_df = pd.DataFrame({
        'proteins': cco_proteins,
        'sequences': cco_sequences,
        'prop_annotations': cco_annotations,
        'embeddings': cco_embeddings
    })
    return bpo_df, mfo_df, cco_df


if __name__ == '__main__':
    args = parser.parse_args()
    main(args.train_file, args.test_file, args.terms_file, args.go_file, args.out_path)
