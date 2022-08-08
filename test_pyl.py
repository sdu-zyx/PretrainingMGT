import pytorch_lightning as pl
print("hello")
hop_sampling_sizes = [16, 8, 4]
for k, sample_size in enumerate(hop_sampling_sizes, start=1):
    print(k, sample_size)

import torch
max_num_ctx_neigh = 15
num_ctx_nodes = 10
attention_mask = torch.zeros(max_num_ctx_neigh + 1, dtype=torch.float32)
attention_mask[: num_ctx_nodes + 1] = 1
print(attention_mask)

num_pairs = 10
for i, num in enumerate(num_pairs):
    print(i, num)