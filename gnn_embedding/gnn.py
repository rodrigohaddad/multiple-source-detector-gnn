import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv, GraphConv

from gnn_embedding.operator import MsgConv


class SAGE(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int,
                 num_layers: int):
        super(SAGE, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else hidden_channels
            self.convs.append(MsgConv(in_channels, hidden_channels, aggr='mean'))

    def forward(self, x, adjs, edge_weight):
        for i, (edge_index, e_id, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x.to(torch.float), x_target.to(torch.float)),
                              edge_index, edge_weight[e_id])
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
        return x

    def full_forward(self, x, edge_index, edge_weight):
        for i, conv in enumerate(self.convs):
            x = conv(x.to(torch.float), edge_index, edge_weight)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
        return x

    @staticmethod
    def loss_fn(out):
        out, pos_out, neg_out, _ = out.split(out.size(0) // 3)

        pos_loss = F.logsigmoid(torch.inner(out, pos_out)).mean()
        neg_loss = F.logsigmoid(-torch.inner(out, neg_out)).mean()
        return -pos_loss - neg_loss

    def fit(self, data, device, epochs, train_loader):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        x, edge_index = data.x.to(device), data.edge_index.to(device)
        self.train()
        val = 0
        # data.val_mask

        for epoch in range(epochs + 1):
            total_loss = 0
            for batch_size, n_id, adjs in train_loader:
                # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
                adjs = [adj.to(device) for adj in adjs]
                optimizer.zero_grad()

                # Escolha dos elementos que compõem o loss. Positivos devem ser "bons" exemplos, e negativos o contrário.
                # Um bom exemplo seria um vizinho
                # Aleatório que não é vizinho e nem vizinho de vizinho.
                # Distribuição pode ser uma uniforme dos vértices que estão a distância >= 3

                out = self(x[n_id], adjs, data.edge_weight)
                loss = self.loss_fn(out)
                loss.backward()
                optimizer.step()

                total_loss += float(loss) * out.size(0)
                val += torch.sum(torch.abs(out))

            if epoch % 10 == 0:
                print(total_loss / data.num_nodes)
                print(f'Epoch {epoch:>3} | Train Loss: {total_loss / len(data):.3f}')

        print('Done')
