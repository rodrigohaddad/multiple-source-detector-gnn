import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils import class_weight
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import accuracy


class SAGESupervised(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int,
                 num_layers: int):
        super(SAGESupervised, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else hidden_channels
            self.convs.append(SAGEConv(in_channels, hidden_channels))

        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=0.001,
                                          weight_decay=5e-4)

    def forward(self, x, adjs):
        for i, (edge_index, e_id, size) in enumerate(adjs):
            x_target = x[:size[1]]
            x = self.convs[i]((x.to(torch.float), x_target.to(torch.float)),
                              edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=0.2, training=self.training)
        return x, F.log_softmax(x, dim=1)

    # def full_forward(self, x, edge_index):
    #     for i, conv in enumerate(self.convs):
    #         x = conv(x.to(torch.float), edge_index)
    #         if i != self.num_layers - 1:
    #             x = x.relu()
    #             x = F.dropout(x, p=0.2, training=self.training)
    #     return x

    def fit(self, data, device, epochs, train_loader):
        x, edge_index, y = data.x.to(device), data.edge_index.to(device), data.y.to(device)
        cl_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y.numpy())
        criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(cl_weights, dtype=torch.float))
        optimizer = self.optimizer
        self.train()
        # data.val_mask

        for epoch in range(epochs + 1):
            acc = 0
            total_loss = 0
            rounds = 0
            for batch_size, n_id, adjs in train_loader:
                adjs = [adj.to(device) for adj in adjs]
                optimizer.zero_grad()

                _, out = self(x[n_id], adjs)
                loss = criterion(out, y[n_id[:batch_size]])
                total_loss += float(loss)
                # print(f'{sum(out.argmax(dim=1))} {sum(y[n_id][:batch_size])}')
                acc += accuracy(out.argmax(dim=1), y[n_id][:batch_size])
                rounds += 1

                loss.backward()
                optimizer.step()

            if epoch % 10 == 0:
                print(f'Epoch {epoch:>3} | Train Loss: {total_loss/rounds:.3f} '
                      f'| Train Acc: {acc/rounds * 100:>6.3f}% ')
        print('Done')
