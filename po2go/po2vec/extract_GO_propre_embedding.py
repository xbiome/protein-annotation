import numpy as np
import pandas as pd
import torch
import pickle

model = torch.load("./models/PO2Vec/part_label_400.pth", map_location="cpu")
proj_con = model["embedding"]

# The file contains all terms you need
with open("./data/terms_all.pkl", "rb") as fd:
    terms_all = pickle.load(fd)
    terms_all = list(terms_all["terms"])

terms_emb_dict = {}
for i in range(len(terms_all)):
    terms_emb_dict[terms_all[i]] = proj_con[i]

for ont in {'cc', 'mf', 'bp'}:
    terms_df = pd.read_pickle("data/{}o_terms.pkl".format(ont))
    terms_emb = np.zeros((len(terms_df), 256), dtype=np.float32)
    # Terms of annotations
    ont_terms = list(terms_df['terms'])

    for i in range(len(ont_terms)):
        terms_emb[i] = terms_emb_dict[ont_terms[i]]
    with open("data/terms_emb/order_{}o_terms_emb.pkl".format(ont), 'wb') as fd:
        pickle.dump(terms_emb, fd)

