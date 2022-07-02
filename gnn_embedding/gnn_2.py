from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GraphConv
from torch_geometric.utils import accuracy


class GraphSAGE2(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.sage1 = GraphConv(dim_in, dim_h, aggr='mean')
        # self.sage2 = GraphConv(dim_h, dim_h, aggr='mean')
        self.sage3 = GraphConv(dim_h, dim_out, aggr='mean')
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=0.01,
                                          weight_decay=5e-4)

    def forward(self, x, edge_index):
        h = self.sage1(x.to(torch.float), edge_index)
        h = torch.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)
        # h = self.sage2(h, edge_index)
        # h = torch.relu(h)
        # h = F.dropout(h, p=0.5, training=self.training)
        h = self.sage3(h, edge_index)
        return h, F.log_softmax(h, dim=1)

    def fit(self, data, epochs):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = self.optimizer
        x = torch.arange(data.num_nodes-1)

        data.train_mask, data.test_mask = train_test_split(x, test_size=0.33, shuffle=True)
        data.train_mask, data.val_mask = train_test_split(data.train_mask, test_size=0.33)

        train_loader = NeighborLoader(
            data,
            num_neighbors=[15, 15, 10],
            batch_size=16,
            input_nodes=data.train_mask,
        )

        self.train()
        for epoch in range(epochs + 1):
            total_loss = 0
            acc = 0
            val_loss = 0
            val_acc = 0

            # Train on batches
            for batch in train_loader:
                optimizer.zero_grad()
                _, out = self(batch.x, batch.edge_index)
                loss = criterion(out[batch.train_mask], batch.y[batch.train_mask])
                total_loss += loss
                acc += accuracy(out[batch.train_mask].argmax(dim=1),
                                batch.y[batch.train_mask])
                loss.backward()
                optimizer.step()

                # Validation
                val_loss += criterion(out[batch.val_mask], batch.y[batch.val_mask])
                val_acc += accuracy(out[batch.val_mask].argmax(dim=1),
                                    batch.y[batch.val_mask])

            # Print metrics every 10 epochs
            if epoch % 10 == 0:
                print(f'Epoch {epoch:>3} | Train Loss: {total_loss / len(train_loader):.3f} '
                      f'| Train Acc: {acc / len(train_loader) * 100:>6.2f}% | Val Loss: '
                      f'{val_loss / len(train_loader):.2f} | Val Acc: '
                      f'{val_acc / len(train_loader) * 100:.2f}%')
