import torch
import torch.nn.functional as F
import torch.nn as nn

from torch_geometric.nn import SAGEConv


# Unsupervised, untransformed graph
class UUSAGE(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out, n_layers):
        super().__init__()
        self.n_layers = n_layers
        self.convs = nn.ModuleList()

        for i in range(n_layers):
            in_channels = dim_in if i == 0 else dim_h
            out_channels = dim_out if i == n_layers - 1 else dim_h
            self.convs.append(SAGEConv(in_channels, out_channels))

        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=0.001,
                                          weight_decay=5e-4)

    def forward(self, x, adjs):
        for i, (conv, info) in enumerate(zip(self.convs, adjs)):
            edge_index, e_id, size = info
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = conv((x.to(torch.float), x_target.to(torch.float)),
                     edge_index)
            if i != self.n_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=0.2, training=self.training)
        return F.log_softmax(x, dim=1), x

    def full_forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x.to(torch.float), edge_index)
            if i != self.n_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=0.2, training=self.training)
        return F.log_softmax(x, dim=1), x

    @staticmethod
    def loss_fn(out):
        out, pos_out, neg_out = out.split(out.size(0) // 3)
        pos_loss = F.logsigmoid(torch.inner(out, pos_out)).mean()
        neg_loss = F.logsigmoid(-torch.inner(out, neg_out)).mean()

        return -pos_loss - neg_loss

    def fit(self, data, device, epochs, train_loader):
        x, edge_index = data.x.to(device), data.edge_index.to(device)
        optimizer = self.optimizer
        self.train()
        val = 0

        self.train()
        for epoch in range(epochs + 1):
            total_loss = 0
            for batch_size, n_id, adjs in train_loader:
                # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
                adjs = [adj.to(device) for adj in adjs]
                optimizer.zero_grad()

                _, out = self(x[n_id], adjs)
                loss = self.loss_fn(out)
                loss.backward()
                optimizer.step()

                total_loss += float(loss)
                val += torch.sum(torch.abs(out))

            if epoch % 10 == 0:
                print(f'Epoch {epoch:>3} | Train Loss: {total_loss / len(train_loader):.3f}')
        print("Done graph training")
