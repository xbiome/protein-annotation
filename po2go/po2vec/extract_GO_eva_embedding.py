import torch
import pickle


model = torch.load("./models/weight_triple/part_order_400.pth", map_location="cpu")
proj_con = model["embedding"].numpy()

# The file contains all terms you need
with open("./data/terms_all.pkl", "rb") as fd:
    terms_all = pickle.load(fd)
    terms_all = list(terms_all["terms"])

terms_emb_dict = {}
for i in range(len(terms_all)):
    terms_emb_dict[terms_all[i]] = proj_con[i]

with open('./data/PO2Vec/order_emb.pkl', 'wb') as fd:
    pickle.dump(terms_emb_dict, fd)

