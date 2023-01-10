import numpy as np
import pandas as pd
import argparse


parser = argparse.ArgumentParser(description='seperate swissprot data into train and test sets.',
                                 add_help=False)

parser.add_argument('--swissprot-file',
                    '-sf',
                    default='data/swissprot.pkl',
                    type=str,
                    help='input protein file in pkl format')
parser.add_argument('--split-ratio',
                    '-s',
                    default=0.9,
                    type=float,
                    help='input protein file in pkl format')
parser.add_argument('--train-data',
                    '-train',
                    default='data/train_data_embedding.pkl',
                    type=str,
                    help='output train data file in pkl format')
parser.add_argument('--test-data',
                    '-test',
                    default='data/test_data_embedding.pkl',
                    type=str,
                    help='output test data file in pkl format')


def main(swissprot_path, split, train_data_path, test_data_path):
    # train data : test data = 90% : 10%
    df_train_data, df_test_data = load_data_to_df_sets(swissprot_path, split, random_split=True)
    df_train_data.to_pickle(train_data_path)
    df_test_data.to_pickle(test_data_path)

# load DataFrame file and split it into two sets


def load_data_to_df_sets(data_file, split=0.9, random_split=True):
    # load DataFrame file
    df = pd.read_pickle(data_file)
    # total number
    n = len(df)
    # index to split
    split_idx = int(n * split)
    # shuffle index
    index = np.arange(n)
    if random_split:
        np.random.seed(seed=0)
        np.random.shuffle(index)
    # Split into set1 and set2
    set1_df = df.iloc[index[:split_idx]]
    set2_df = df.iloc[index[split_idx:]]

    return set1_df, set2_df


if __name__ == '__main__':
    args = parser.parse_args()
    main(args.swissprot_file, args.split_ratio, args.train_data, args.test_data)
