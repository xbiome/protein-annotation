import gc
import os
import random
from typing import Dict

import esm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from partialgo.utils.constant import DEFAULT_ESM_MODEL, ESM_LIST

class EsmDataset(Dataset):
    """ESMDataset."""
    def __init__(self,
                 test_df: pd.DataFrame,
                 model_dir: str = 'esm1b_t33_650M_UR50S',
                 max_length: int = 1024,
                 truncate: bool = True,
                 random_crop: bool = False):
        super().__init__()
        self.df = test_df
        self.max_length = max_length
        self.truncate = truncate
        self.random_crop = random_crop

        if model_dir not in ESM_LIST:
            print(
                f"Model dir '{model_dir}' not recognized. Using '{DEFAULT_ESM_MODEL}' as default"
            )
            model_dir = DEFAULT_ESM_MODEL

        self.is_msa = 'msa' in model_dir

        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(
            model_dir)
        self.batch_converter = self.alphabet.get_batch_converter()

    @property
    def vocab_size(self) -> int:
        """Returns the whole vocabulary size."""
        return len(list(self.alphabet.tok_to_idx.keys()))

    @property
    def mask_token(self) -> str:
        """Representation of the mask token (as a string)"""
        return self.alphabet.all_toks[self.alphabet.mask_idx]  # "<mask>"

    @property
    def pad_token(self) -> str:
        """Representation of the pad token (as a string)"""
        return self.alphabet.all_toks[self.alphabet.padding_idx]  # "<pad>"

    @property
    def begin_token(self) -> str:
        """Representation of the beginning of sentence token (as a string)"""
        return self.alphabet.all_toks[self.alphabet.cls_idx]  # "<cls>"

    @property
    def end_token(self) -> str:
        """Representation of the end of sentence token (as a string)"""
        return self.alphabet.all_toks[self.alphabet.eos_idx]  # "<eos>"

    @property
    def does_end_token_exist(self) -> bool:
        """Returns true if a end of sequence token exists."""
        return self.alphabet.append_eos

    @property
    def token_to_id(self):
        """Returns a function which maps tokens to IDs."""
        return lambda x: self.alphabet.tok_to_idx[x]

    def free_memory(self, esm_model):
        del esm_model
        gc.collect()
        print('Delete the esm model, free memory!')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sequence = self.df.sequences[idx]
        if self.truncate:
            sequence = sequence[:self.max_length - 1]
        length = len(sequence)
        return sequence, self.mask_length(length)

    def mask_length(self, length:int):
        masked_length = torch.cat((torch.ones((1, length)), torch.zeros((1, self.max_length-1-length))), dim=1)
        return masked_length

    def collate_fn(self, examples) -> Dict[str, torch.Tensor]:
        """Function to transform tokens string to IDs; it depends on the model
        used."""
        sequences_list = [ex[0] for ex in examples]
        masked_length_list = [ex[1] for ex in examples]

        if self.is_msa:
            _, _, all_tokens = self.batch_converter(sequences_list)
        else:
            _, _, all_tokens = self.batch_converter([
                ('', sequence) for sequence in sequences_list
            ])

        # The model is trained on truncated sequences and passing longer ones in at
        # infernce will cause an error. See https://github.com/facebookresearch/esm/issues/21
        if self.truncate:
            all_tokens = all_tokens[:, :self.max_length]

        if all_tokens.shape[1] < 1024:
            tmp = torch.ones((all_tokens.shape[0], 1024 - all_tokens.shape[1]))
            all_tokens = torch.cat([all_tokens, tmp], dim=1)
        all_tokens = all_tokens.int()
        all_tokens = all_tokens.to('cpu')
        encoded_inputs = {
            'input_ids': all_tokens,
        }
        encoded_inputs['masked_lengths'] = torch.cat(masked_length_list, dim=0)

        return encoded_inputs

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    # test CAFA3 dataset
    test_df = pd.read_pickle('../../data/cafa3/mfo/mfo_train_data.pkl')
    mfo_dataset = EsmDataset(test_df)
    mfo_loader = DataLoader(mfo_dataset,
                            batch_size=8,
                            collate_fn=mfo_dataset.collate_fn)
    for index, batch in enumerate(mfo_loader):
        for key, val in batch.items():
            # print(key, val.shape)
            print(batch['masked_lengths'].sum(1))
        if index > 20:
            break
