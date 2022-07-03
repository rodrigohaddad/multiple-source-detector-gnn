import torch
from torch_geometric.utils import accuracy


def test(model, data):
    model.eval()
    try:
        data.test_mask
    except:
        data.test_mask = torch.arange(data.num_nodes - 1)
    _, out = model(data.x, data.edge_index)
    acc = accuracy(out.argmax(dim=1)[data.test_mask], data.y[data.test_mask])
    return acc
