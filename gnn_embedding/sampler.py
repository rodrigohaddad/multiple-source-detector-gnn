import torch
from torch_cluster import random_walk

from torch_geometric.loader import NeighborSampler

from utils.constants import DEVICE


class PosNegSampler(NeighborSampler):
    def sample(self, batch):
        batch = torch.tensor(batch)
        row, col, _ = self.adj_t.coo()

        # rowptr, col_2, __ = self.adj_t.csr()
        # u = 2
        # nbrs = col_2[rowptr[u]: rowptr[u + 1]]
        # print(nbrs)

        pos_batch = random_walk(row, col, batch, walk_length=1,
                                coalesced=False)
        pos_batch = pos_batch[:, 1]

        # neg_batch = torch.empty((batch.numel(), ), dtype=torch.long, device=DEVICE)
        # for i in range(batch.numel()):
        #     validating_element = False
        #     while not validating_element:
        #         element = torch.randint(0, self.adj_t.size(1), (1,), dtype=torch.long)
        #         if element not in batch:
        #             validating_element = True
        #     neg_batch[i] = element

        neg_batch = torch.randint(0, self.adj_t.size(1), (batch.numel(), ),
                                  dtype=torch.long)

        batch = torch.cat([batch, pos_batch, neg_batch], dim=0)
        return super().sample(batch)
