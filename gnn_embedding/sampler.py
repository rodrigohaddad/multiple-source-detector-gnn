import collections

import torch
from torch_cluster import random_walk

from torch_geometric.loader import NeighborSampler, NeighborLoader


class PosNegSampler(NeighborSampler):
    def sample(self, batch):
        batch = torch.tensor(batch)
        row, col, _ = self.adj_t.coo()

        pos_batch = random_walk(row, col, batch, walk_length=1,
                                coalesced=False)
        pos_batch = pos_batch[:, 1]

        neg_batch = torch.randint(0, self.adj_t.size(1), (batch.numel(), ),
                                  dtype=torch.long)

        batch = torch.cat([batch, pos_batch, neg_batch], dim=0)
        # aa = [item for item, count in collections.Counter(batch.tolist()).items() if count > 1]
        return super().sample(batch)

