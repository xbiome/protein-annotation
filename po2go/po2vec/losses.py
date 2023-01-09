from __future__ import print_function

import torch
import torch.nn as nn


class ConPairLoss(nn.Module):
    def __init__(self, temperature=0.1, neg_num=80):
        super().__init__()
        self.temperature = temperature
        self.neg_num = neg_num

    def forward(self, features, multi_rate):
        # features shape i.e. (batchsize, samplesize, dim)
        o = features.narrow(1, 1, features.size(1) - 1)
        s = features.narrow(1, 0, 1)
        s = torch.transpose(s, 1, 2)
        similarity = o @ s
        similarity = similarity.squeeze()
        similarity = torch.exp(
            similarity /
            self.temperature)  # shape i.e. (barchsize, samplesize)
        s = similarity[:, 0]
        negs = similarity[:, 1:] * multi_rate * self.neg_num / 2
        temp = s / (s + torch.sum(negs, dim=-1))
        temp = 0 - torch.log(temp)
        return torch.mean(temp)
