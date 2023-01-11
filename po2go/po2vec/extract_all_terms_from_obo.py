import argparse
import pandas as pd
from po2go.po2vec.utils import Ontology


parser = argparse.ArgumentParser(description='extract all terms in a go.obo file.',
                                 add_help=False)

parser.add_argument('--go-file',
                    '-g',
                    default='data/go.obo',
                    type=str,
                    help='go file downloaded from Gene Ontology website')
parser.add_argument('--out-file',
                    '-o',
                    default='data/terms_all.pkl',
                    type=str,
                    help='terms stored as a DataFrame in pkl format')


def main(go_file, out_file):
    go = Ontology(go_file, with_rels=True, include_alt_ids=False)
    df_terms = pd.DataFrame({'terms': go.ont.keys()})
    df_terms.to_pickle(out_file)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args.go_file, args.out_file)
