import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    """ESMDataset."""

    def __init__(self,
                 label_map: dict,
                 root_path: str = 'dataset/',
                 file_name: str = 'xxx.pkl'):
        super().__init__()
        self.data_path = os.path.join(root_path, file_name)
        self.embeddings, self.labels = self.load_dataset(self.data_path)
        self.terms_dict = label_map
        self.num_classes = len(self.terms_dict)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        label_list = self.labels[idx]
        multilabel = [0] * self.num_classes
        for t_id in label_list:
            if t_id in self.terms_dict:
                label_idx = self.terms_dict[t_id]
                multilabel[label_idx] = 1

        embeddings = torch.from_numpy(np.array(embedding, dtype=np.float32))
        labels = torch.from_numpy(np.array(multilabel))
        encoded_inputs = {'embeddings': embeddings, 'labels': labels}
        return encoded_inputs

    def load_dataset(self, data_path):
        df = pd.read_pickle(data_path)
        embeddings = list(df['embeddings'])
        label = list(df['prop_annotations'])
        assert len(embeddings) == len(label)
        return embeddings, label
