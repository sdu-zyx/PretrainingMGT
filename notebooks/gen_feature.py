
import gzip
import json
import os
from collections import Counter
from datetime import datetime
from functools import partial

import backoff
import joblib
import networkx as nx
import numpy as np
import pandas as pd
import requests
import scipy.sparse as sp
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import Parallel, delayed
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange
from transformers import AutoModel, AutoTokenizer


# VG Dataset
from PMGT_main.pmgt.preprocessing.datasets import AmazonReviewImageDataset, AmazonReviewTextDataset, text_collate_fn

data_dir = "../data/VG"
filename = "Video_Games_5.json.gz"


with gzip.open(os.path.join(data_dir, filename)) as f:
    data = [json.loads(l.strip()) for l in tqdm(f)]

df = pd.DataFrame.from_dict(data)
df['reviewDateTime'] = df['unixReviewTime'].map(lambda x: datetime.fromtimestamp(x))
df = df.sort_values(by='reviewDateTime')
print(len(df))

criterion = datetime(2015, 1, 1, 9)
df1 = df[df['reviewDateTime'] < criterion].reset_index(drop=True)
df2 = df[df['reviewDateTime'] >= criterion].reset_index(drop=True)
print(len(df1))
print(len(df2))

## Download Images

# %%

image_root_path = os.path.join(data_dir, "images")
os.makedirs(image_root_path, exist_ok=True)


# %%

def _giveup(e):
    return str(e) == "404"


@backoff.on_exception(
    backoff.expo,
    (requests.exceptions.RequestException, requests.exceptions.ConnectionError),
    max_time=30,
    max_tries=5,
    giveup=_giveup,
)
def download_image(filepath, image_url):
    if os.path.exists(filepath):
        return False

    try:
        r = requests.get(image_url, stream=True)
    except requests.exceptions.MissingSchema:
        return False

    if r.status_code == 404:
        return False
    elif r.status_code != 200:
        raise requests.exceptions.RequestException(r.status_code)

    with open(filepath, "wb") as f:
        for chunk in r.iter_content(1024):
            f.write(chunk)

    return True


download_list = []
counter = Counter()

for index, row in df1[~pd.isna(df1["image"])].iterrows():
    for i, image_url in enumerate(row["image"]):
        ext = os.path.splitext(image_url)[1]
        item_id = row["asin"]
        filepath = os.path.join(image_root_path, item_id, f"{counter[item_id]}{ext}")
        counter[item_id] += 1
        download_list.append((filepath, image_url))

        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

results = Parallel(n_jobs=50, prefer="threads")(
    delayed(download_image)(f, u) for f, u in tqdm(download_list)
)

print(len(download_list))
print(len(df1["asin"].unique()))
print(len(next(os.walk(image_root_path))[1]))
print("download over")

# %% md

# Extract Visual Features

# %%

os.environ['CUDA_VISIBLE_DEVICES'] = "1"

# %%

model = timm.create_model("inception_v4", pretrained=True)
config = resolve_data_config({}, model=model)
transform = create_transform(**config)
dataset = AmazonReviewImageDataset(
    image_root_path, transforms=transform, item_ids=df1["asin"].unique()
)

dataloader = DataLoader(dataset, batch_size=32, num_workers=8)

model.cuda()
model.eval()

visual_feats = []
for batch_x in tqdm(dataloader, total=len(dataloader)):
    batch_x = batch_x.cuda()
    with torch.no_grad():
        feat = model.global_pool(model.forward_features(batch_x))
        visual_feats.append(feat.cpu())

visual_feats = torch.cat(visual_feats)

item_visual_feats = []
start = 0
for num in tqdm(dataset.num_images.values()):
    end = start + num
    item_visual_feats.append(visual_feats[start:end].mean(dim=0))
    start = end
item_visual_feats = torch.stack(item_visual_feats).numpy()
item_mapping = np.array([item_id for item_id in dataset.num_images.keys()])

np.savez(
    os.path.join(data_dir, "visual_feats.npz"),
    feats=item_visual_feats,
    mapping=item_mapping,
)
print("save visual over")

# %% md

## Extract Textual Features

# %%

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

# %%

review_text = (
    df1[~pd.isna(df1["reviewText"])]
        .groupby("asin")
        .apply(lambda r: r["reviewText"].values)
)
review_text = review_text.to_dict()

dataset = AmazonReviewTextDataset(review_text)
model_name = "bert-base-uncased"

model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.cuda()
model.eval()

dataloader = DataLoader(
    dataset,
    batch_size=64,
    num_workers=16,
    collate_fn=partial(text_collate_fn, tokenizer=tokenizer),
)

text_feats = []

for batch_x in tqdm(dataloader, total=len(dataloader)):
    batch_x = {k: v.cuda() for k, v in batch_x.items()}
    with torch.no_grad():
        text_feats.append(model(**batch_x)[0][:, 0].cpu())

text_feats = torch.cat(text_feats)

item_textual_feats = []
start = 0
for num in tqdm(dataset.num_texts.values()):
    end = start + num
    item_textual_feats.append(text_feats[start:end].mean(dim=0))
    start = end
item_textual_feats = torch.stack(item_textual_feats).numpy()
item_mapping = np.array([item_id for item_id in dataset.num_texts.keys()])

np.savez(
    os.path.join(data_dir, "textual_feats.npz"),
    feats=item_textual_feats,
    mapping=item_mapping,
)

print("save text over")

# %% md

## Construct Product Graph

# %%

graph_data = []
users_per_item = df1.groupby(by="asin").apply(lambda r: set(r["reviewerID"].unique()))
item_ids = df1["asin"].unique()

item_encoder = LabelEncoder().fit(item_ids)
user_encoder = LabelEncoder().fit(df1["reviewerID"].unique())

item_user_mat = sp.dok_matrix(
    (len(item_encoder.classes_), len(user_encoder.classes_)), dtype=np.int32
)

item_to_idx = {v: i for i, v in enumerate(item_encoder.classes_)}
user_to_idx = {v: i for i, v in enumerate(user_encoder.classes_)}

for item in tqdm(item_ids):
    item_id = item_to_idx[item]
    user_ids = [user_to_idx[u] for u in users_per_item[item]]
    item_user_mat[item_id, user_ids] = 1

item_user_mat_csr = item_user_mat.tocsr()
item_item_mat = item_user_mat_csr @ item_user_mat_csr.T
item_item_mat.setdiag(0)
item_item_mat.eliminate_zeros()

graph_data = []
for i, row in enumerate(tqdm(item_item_mat, total=item_item_mat.shape[0])):
    for j, r in zip(row.indices, row.data):
        if r >= 3:
            graph_data.append((item_encoder.classes_[i], item_encoder.classes_[j], r))

G = nx.Graph()
G.add_weighted_edges_from(graph_data)

for u, v, w in tqdm(G.edges.data("weight")):
    w = (np.log(w) + 1) / (np.log(np.sqrt(G.degree[u] * G.degree[v])) + 1)
    G.edges[u, v]["weight"] = w

nx.write_gpickle(G, os.path.join(data_dir, "graph.gpickle"))

print(G.number_of_nodes())
print(G.number_of_edges())

# %% md

## Node Encoder

# %%

node_encoder = LabelEncoder().fit(list(G.nodes.keys()))
joblib.dump(node_encoder, os.path.join(data_dir, "node_encoder"))

# %% md

## Filter Out Interactions

# %%

df3 = df2[df2['asin'].isin(G.nodes.keys())].reset_index(drop=True)

# %% md

## User & Item Encoder

# %%

user_encoder = LabelEncoder().fit(df3['reviewerID'].unique())
item_encoder = LabelEncoder().fit(df3['asin'].unique())

joblib.dump(user_encoder, os.path.join(data_dir, 'user_encoder'))
joblib.dump(item_encoder, os.path.join(data_dir, 'item_encoder'))

# %% md

## Split Train & Test

# %%

random_state = np.random.RandomState(2022)
train_df, test_df = train_test_split(df3, test_size=0.2, random_state=random_state)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)
train_df.to_json(os.path.join(data_dir, 'train.json'), date_format='iso')
test_df.to_json(os.path.join(data_dir, 'test.json'), date_format='iso')

# %% md

## Save Initial Feature Embeddings

# %%

node_encoder = joblib.load(os.path.join(data_dir, "node_encoder"))

with np.load(os.path.join(data_dir, "visual_feats.npz"), allow_pickle=True) as npz:
    visual_feats = npz["feats"]
    visual_feats_mapping = npz["mapping"]

with np.load(os.path.join(data_dir, "textual_feats.npz"), allow_pickle=True) as npz:
    textual_feats = npz["feats"]
    textual_feats_mapping = npz["mapping"]


def get_feat_init_emb(node_size, items, feats, item_to_idx):
    feat_init_emb = np.empty((node_size + 2, feats.shape[1]), dtype=np.float32)
    feat_init_emb[0] = np.zeros_like(feat_init_emb[0])  # 0 for padding
    feat_init_emb[1] = np.zeros_like(feat_init_emb[1])  # 1 for masking

    for i, item in enumerate(items, start=2):
        if item not in item_to_idx:
            feat_init_emb[i] = np.random.normal(size=feats.shape[1])
        else:
            feat_init_emb[i] = feats[item_to_idx[item]]

    return feat_init_emb


node_size = len(node_encoder.classes_)
item_to_idx = {item: i for i, item in enumerate(visual_feats_mapping)}
visual_init_emb = get_feat_init_emb(
    node_size, node_encoder.classes_, visual_feats, item_to_idx
)

item_to_idx = {item: i for i, item in enumerate(textual_feats_mapping)}
textual_init_emb = get_feat_init_emb(
    node_size, node_encoder.classes_, textual_feats, item_to_idx
)

np.save(
    os.path.join(data_dir, "visual_init_emb.npy"),
    visual_init_emb,
)

np.save(
    os.path.join(data_dir, "textual_init_emb.npy"),
    textual_init_emb,
)

# %% md

# %% md

# Calculate Embedding Size

# %%

factor_num = 16
num_layers = 2
emb_size = factor_num * 2 ** (num_layers - 1)
print(emb_size)
