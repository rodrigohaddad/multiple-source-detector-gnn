import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.loader import NeighborSampler, NeighborLoader
from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool, GraphConv
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import accuracy

DEVICE = torch.device('cpu' if torch.cuda.is_available() else 'cpu')


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_layers):
        super(GraphSAGE, self).__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList().to(DEVICE)
        # self.optimizer = torch.optim.Adam(self.parameters(),
        #                                   lr=0.01,
        #                                   weight_decay=5e-4)
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            self.convs.append(GraphConv(in_channels, out_channels, aggr='mean').to(DEVICE))

    def forward(self, x, adjs, edge_weight):
        for i, (edge_index, e_id, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x.to(torch.float), x_target.to(torch.float)), edge_index.to(DEVICE), edge_weight[e_id])
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
        return x, F.log_softmax(x, dim=1)

    # def forward(self, x, edge_index):
    #     h = self.sage1(x, edge_index)
    #     h = torch.relu(h)
    #     h = F.dropout(h, p=0.5, training=self.training)
    #     h = self.sage2(h, edge_index)
    #     return h, F.log_softmax(h, dim=1)

    # def full_forward(self, x, edge_index):
    #     for i, conv in enumerate(self.convs):
    #         x = conv(x, edge_index)
    #         if i != self.num_layers - 1:
    #             x = x.relu()
    #             x = F.dropout(x, p=0.5, training=self.training)
    #     return x

    def fit(self, data, epochs):
        criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
        # optimizer = self.optimizer
        data.train_mask = torch.tensor(data=[0, 500], dtype=torch.float).to(DEVICE)
        train_loader = NeighborSampler(data.edge_index, sizes=[10, 10], batch_size=256,
                                       shuffle=True, num_nodes=data.num_nodes)
        train_loader.edge_index = train_loader.edge_index.to(DEVICE)

        # train_loader = NeighborLoader(
        #     data,
        #     num_neighbors=[5, 10],
        #     batch_size=16,
        #     input_nodes=data.train_mask,
        # )

        self.train()
        for epoch in range(epochs + 1):
            total_loss = 0
            acc = 0
            val_loss = 0
            val_acc = 0

            # Train on batches
            # for batch in train_loader_2:
            for batch_size, n_id, adjs in train_loader:
                _, out = self(data.x[n_id], adjs, data.edge_weight)
                print("a")
                # optimizer.zero_grad()
                # _, out = self(batch.x, batch.edge_index)
                # _, out = self(batch.x, batch.edge_index)
                # loss = criterion(out[batch.train_mask], batch.y[batch.train_mask])
                # total_loss += loss
                # acc += accuracy(out[batch.train_mask].argmax(dim=1),
                #                 batch.y[batch.train_mask])
                # loss.backward()
                # optimizer.step()

                # Validation
                # val_loss += criterion(out[batch.val_mask], batch.y[batch.val_mask])
                # val_acc += accuracy(out[batch.val_mask].argmax(dim=1),
                #                     batch.y[batch.val_mask])

            # Print metrics every 10 epochs
            if epoch % 10 == 0:
                print(f'Epoch {epoch:>3} | Train Loss: {total_loss / len(train_loader):.3f} '
                      f'| Train Acc: {acc / len(train_loader) * 100:>6.2f}% | Val Loss: '
                      f'{val_loss / len(train_loader):.2f} | Val Acc: '
                      f'{val_acc / len(train_loader) * 100:.2f}%')

    def __repr__(self):
        return self.__class__.__name__


# class myGNN(torch.nn.Module):
#     def __init__(self, dataset, num_layers, hidden):
#         super().__init__()
#         GraphConv()
#         self.conv1 = SAGEConv(dataset.num_features, hidden)
#         self.convs = torch.nn.ModuleList()
#         for i in range(num_layers - 1):
#             self.convs.append(SAGEConv(hidden, hidden))
#         self.lin1 = Linear(hidden, hidden)
#         self.lin2 = Linear(hidden, dataset.num_classes)
#
#     def reset_parameters(self):
#         self.conv1.reset_parameters()
#         for conv in self.convs:
#             conv.reset_parameters()
#         self.lin1.reset_parameters()
#         self.lin2.reset_parameters()
#
#     def forward(self, data):
#         x, edge_index, batch = data.x, data.edge_index, data.batch
#         x = F.relu(self.conv1(x, edge_index))
#         for conv in self.convs:
#             x = F.relu(conv(x, edge_index))
#         x = global_mean_pool(x, batch)
#         x = F.relu(self.lin1(x))
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = self.lin2(x)
#         return F.log_softmax(x, dim=-1)
#
#     def __repr__(self):
#         return self.__class__.__name__
