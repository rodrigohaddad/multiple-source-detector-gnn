import numpy as np
import torch
from sklearn.utils import class_weight
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.utils import accuracy


# Supervised, not transformed graph
class SUSAGEBin(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out, n_layers, aggr):
        super().__init__()
        self.aggr = aggr
        self.n_layers = n_layers
        self.convs = nn.ModuleList()

        for i in range(n_layers):
            in_channels = dim_in if i == 0 else dim_h
            out_channels = dim_out if i == n_layers - 1 else dim_h
            conv = SAGEConv(in_channels, out_channels)
            conv.aggr = self.aggr
            self.convs.append(conv)

        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=0.001,
                                          weight_decay=5e-4)

    def forward(self, x, edge_index):
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if i != self.n_layers - 1:
                h = torch.relu(h)
                h = F.dropout(h, p=0.1, training=self.training)
        return h, torch.sigmoid(h)

    def fit(self, data, epochs, train_loader):
        cl_weights = class_weight.compute_class_weight('balanced',
                                                       classes=np.unique(data.y),
                                                       y=data.y.numpy())

        weights = data.y*max(cl_weights)*10
        weights[weights == 0] = min(cl_weights)
        weights = weights.unsqueeze(1)

        criterion = torch.nn.BCELoss(weight=weights)
        optimizer = self.optimizer

        self.train()
        for epoch in range(epochs + 1):
            total_loss = 0
            acc = 0

            for batch in train_loader:
                optimizer.zero_grad()
                _, out = self(batch.x, batch.edge_index)
                unsqueezed = batch.y.unsqueeze(1)
                if len(unsqueezed) != 1500:
                    print("aqui")
                loss = criterion(out, unsqueezed)
                total_loss += loss
                # acc += accuracy(out.argmax(dim=1),
                #                 batch.y)

                loss.backward()
                optimizer.step()

            if epoch % 10 == 0:
                # print(f'N pred sources: {int(sum(out.argmax(dim=1)))}')
                print(f'N pred sources: {(0.5 < out).sum()}')
                print(f'Epoch {epoch:>3} | Train Loss: {total_loss / len(train_loader):.3f} '
                      f'| Train Acc: {acc / len(train_loader) * 100:>6.2f}%')
        print("Done graph training")
