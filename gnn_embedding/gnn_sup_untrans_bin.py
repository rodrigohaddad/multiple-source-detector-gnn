import numpy as np
import torch
from sklearn.utils import class_weight
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
import torch.nn as nn

from gnn_embedding.early_stopper import EarlyStopper


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
        # return torch.sigmoid(h), h
        return h, torch.sigmoid(h)

    @staticmethod
    def calculate_weights(batch, cl_weights, multiplier):
        weights = batch.y * max(cl_weights) * multiplier
        weights[weights == 0] = min(cl_weights)
        return weights.unsqueeze(1)

    @torch.no_grad()
    def validation(self, criterion, cl_weights_val, val_batch):
        weights_val = self.calculate_weights(val_batch, cl_weights_val, 100)
        criterion.register_buffer('weight', weights_val)
        out, out_sig = self(val_batch.x, val_batch.edge_index)
        unsqueezed = val_batch.y.unsqueeze(1)
        return criterion(out, unsqueezed)

    def fit(self, data, epochs, train_loader, data_val):
        cl_weights = class_weight.compute_class_weight('balanced',
                                                       classes=np.unique(data.y),
                                                       y=data.y.numpy())

        cl_weights_val = class_weight.compute_class_weight('balanced',
                                                           classes=np.unique(data_val.y),
                                                           y=data_val.y.numpy())

        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = self.optimizer

        early_stopper = EarlyStopper(patience=75)

        self.train()
        for epoch in range(epochs + 1):
            total_loss = 0
            val_total_loss = 0
            steps = 0

            for train_batch in train_loader:
                optimizer.zero_grad()
                out, out_sig = self(train_batch.x, train_batch.edge_index)
                unsqueezed = train_batch.y.unsqueeze(1)

                weights = self.calculate_weights(train_batch, cl_weights, 100)
                criterion.register_buffer('weight', weights)
                # criterion.weight = weights

                loss = criterion(out, unsqueezed)
                total_loss += loss
                # acc += accuracy(out.argmax(dim=1),
                #                 batch.y)

                loss.backward()
                optimizer.step()

                # Validation
                val_total_loss += self.validation(criterion, cl_weights_val, data_val)
                steps += 1

            if epoch % 10 == 0:
                # print(f'N pred sources: {int(sum(out.argmax(dim=1)))}')
                # pc = np.percentile(out_sig.detach().numpy(), 98)
                print(np.asarray(data.y == 1).nonzero())
                print(f'N pred sources: {(out_sig.mean() * 1.5 < out_sig).sum()}')
                print(f'Epoch {epoch:>3} | Train Loss: {total_loss / len(train_loader):.3f}')

            early_stopper.register_val_loss(val_total_loss/steps, epoch)
            if early_stopper.should_stop():
                break
        print("Done graph training")
