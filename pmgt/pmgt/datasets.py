"""
Created on 2022/01/08
@author Sangwoo Han
"""
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import scipy.special as ss
import torch


def _sample_context_neigh(
    graph: nx.Graph,
    target_node: int,
    hop_sampling_sizes: List[int],
    max_num_ctx_neigh: int,
) -> Tuple[List[int], int]:
    scores = defaultdict(lambda: 0)
    depth = len(hop_sampling_sizes)

    sampled_nodes = [[target_node]] + [[] for _ in range(depth)]

    for k, sample_size in enumerate(hop_sampling_sizes, start=1):
        for node in sampled_nodes[k - 1]:
            weights = ss.softmax(
                np.asarray([v["weight"] for v in graph[node].values()])
            )
            node_list = np.random.choice(
                graph[node], size=sample_size, replace=True, p=weights
            ).tolist()
            sampled_nodes[k].extend(node_list)

        counter = Counter(sampled_nodes[k])

        for node, freq in counter.items():
            if node == target_node:
                continue
            scores[node] += freq * (depth - k + 1)

    ctx_nodes, _ = zip(*sorted(scores.items(), key=lambda item: item[1], reverse=True))

    ctx_nodes = list(ctx_nodes)

    if len(ctx_nodes) < max_num_ctx_neigh:
        num_ctx_nodes = len(ctx_nodes)
        ctx_nodes += [0] * (max_num_ctx_neigh - len(ctx_nodes))
    else:
        num_ctx_nodes = max_num_ctx_neigh
        ctx_nodes = ctx_nodes[:max_num_ctx_neigh]

    return ctx_nodes, num_ctx_nodes


def _get_attention_mask(
    num_ctx_nodes: int, max_num_ctx_neigh: int
) -> torch.FloatTensor:
    attention_mask = torch.zeros(max_num_ctx_neigh + 1, dtype=torch.float32)
    attention_mask[: num_ctx_nodes + 1] = 1
    return attention_mask


def get_input_tensor(
    graph: nx.Graph,
    target_node: int,
    hop_sampling_sizes: List[int],
    max_num_ctx_neigh: int,
) -> Tuple[torch.LongTensor, torch.FloatTensor]:
    ctx_nodes, num_ctx_nodes = _sample_context_neigh(
        graph, target_node, hop_sampling_sizes, max_num_ctx_neigh
    )

    assert (
        len(ctx_nodes) == max_num_ctx_neigh
    ), f"# of context nodes must be {max_num_ctx_neigh}"

    attention_mask = _get_attention_mask(num_ctx_nodes, max_num_ctx_neigh)
    return torch.LongTensor([target_node] + ctx_nodes), attention_mask


class PMGTDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        graph: nx.Graph,
        node_ids: Optional[np.ndarray] = None,
        max_ctx_neigh: int = 5,
        hop_sampling_sizes: List[int] = [16, 8, 4],
        max_total_samples: int = 10,
        min_neg_samples: int = 5,
        is_training: bool = True,
        is_inference: bool = False,
    ) -> None:
        super().__init__()
        self.graph = graph
        self.node_ids = (
            node_ids
            if node_ids is not None
            # 0 is <pad>
            # 1 is <mask>
            else np.arange(start=2, stop=len(self.graph) + 2)
        )
        self.max_num_ctx_neigh = max_ctx_neigh
        self.hop_sampling_sizes = hop_sampling_sizes
        self.max_total_samples = max_total_samples
        self.min_neg_samples = min_neg_samples
        self.is_training = is_training
        self.is_inference = is_inference

    def __len__(self) -> int:
        return len(self.node_ids)  # num of nodes

    def __getitem__(
        self, idx: int
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        target_node = self.node_ids[idx]

        target_inputs = get_input_tensor(
            self.graph, target_node, self.hop_sampling_sizes, self.max_num_ctx_neigh
        )

        if self.is_inference:
            return (target_inputs,)

        num_neigh_nodes = (
            self.max_total_samples - self.min_neg_samples if self.is_training else 1
        )
        neigh_nodes = self._sample_neigh(target_node, num_neigh_nodes)

        neigh_input_ids, neigh_attn_mask = zip(
            *[
                get_input_tensor(
                    self.graph, n, self.hop_sampling_sizes, self.max_num_ctx_neigh
                )
                for n in neigh_nodes
            ]
        )
        neigh_input_ids = torch.stack(neigh_input_ids)
        neigh_attn_mask = torch.stack(neigh_attn_mask)

        num_neg = (
            max(self.min_neg_samples, self.max_total_samples - len(neigh_nodes))
            if self.is_training
            else 1
        )
        neg_nodes = self._sample_neg(target_node, num_neg)

        neg_input_ids, neg_attn_mask = zip(
            *[
                get_input_tensor(
                    self.graph, n, self.hop_sampling_sizes, self.max_num_ctx_neigh
                )
                for n in neg_nodes
            ]
        )
        neg_input_ids = torch.stack(neg_input_ids)
        neg_attn_mask = torch.stack(neg_attn_mask)

        labels = self._get_label_tensor(neigh_input_ids.size(0), neg_input_ids.size(0))

        pair_inputs = torch.cat([neigh_input_ids, neg_input_ids]), torch.cat(
            [neigh_attn_mask, neg_attn_mask]
        )

        return target_inputs, pair_inputs, labels

    def _sample_neigh(self, target_node: int, max_samples: int) -> List[int]:
        neigh = list(self.graph[target_node].keys())
        num_samples = min(max_samples, len(neigh))
        sampled = np.random.choice(neigh, num_samples, replace=False).tolist()
        return sampled

    def _sample_neg(self, target_node: int, num_samples: int) -> List[int]:
        neg_nodes = []
        for _ in range(num_samples):
            candidate = np.random.randint(len(self.graph)) + 2
            while candidate in self.graph[target_node]:
                candidate = np.random.randint(len(self.graph)) + 2
            neg_nodes.append(candidate)
        return neg_nodes

    def _get_label_tensor(self, num_pos: int, num_neg: int) -> torch.Tensor:
        return torch.FloatTensor([1] * num_pos + [0] * num_neg)


def pmgt_collate_fn(
    batch: Iterable[Tuple[torch.Tensor, ...]]
) -> Dict[str, torch.Tensor]:

    target_inputs = {
        "node_ids": torch.stack([b[0][0] for b in batch]),
        "attention_mask": torch.stack([b[0][1] for b in batch]),
    }

    if len(batch[0]) == 1:
        return target_inputs

    pair_inputs = {
        "node_ids": torch.cat([b[1][0] for b in batch]),
        "attention_mask": torch.cat([b[1][1] for b in batch]),
    }
    num_pairs = torch.LongTensor([len(b[1][0]) for b in batch])

    # pair_inputs = [{"node_ids": b[1][0], "attention_mask": b[1][1]} for b in batch]

    labels = torch.cat([b[2] for b in batch])

    return target_inputs, pair_inputs, num_pairs, labels
