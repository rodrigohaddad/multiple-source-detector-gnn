import torch
import pickle
import numpy as np

from utils.constants import INFECTED_DIR


@torch.no_grad()
def test_embedding(model, data, no_weight):
    model.eval()
    try:
        data.test_mask
    except:
        data.test_mask = torch.arange(data.num_nodes - 1)

    if no_weight:
        _, out = model.full_forward(data.x, data.edge_index)
    else:
        out = model.full_forward(data.x, data.edge_index, data.edge_weight).cpu()

    # _, out = model(data.x, data.edge_index, data.edge_weight)
    # acc = accuracy(out.argmax(dim=1)[data.test_mask], data.y[data.test_mask])
    return out


@torch.no_grad()
def test_pred(model, data, sources):
    model.eval()
    _, out = model(data.x, data.edge_index)
    # out.size(dim=1)
    # y_pred = out.argmax(dim=1)
    # return y_pred

    out_np = torch.clone(out.squeeze(dim=1))
    indices = torch.topk(out_np, sources).indices

    out_np.zero_()
    # y_pred = torch.where(out_np < out_np.mean()*1.9, 0, 1)
    for idx in indices:
        out_np[idx] = 1
    return out_np, indices


def concatenate_sources(file, filename, sources, conj_emb, emb=None):
    if emb is None:
        emb = pickle.load(open(file, 'rb'))
    inf_model = pickle.load(open(f'{INFECTED_DIR}/{filename.split("-")[0]}-infected.pickle', 'rb'))

    infections = list(inf_model.model.status.values())
    emb = torch.column_stack((emb, torch.Tensor(infections)))

    sources = torch.concat((sources, torch.IntTensor(list(inf_model.model.initial_status.values()))))
    conj_emb = torch.concat((conj_emb, emb))

    return conj_emb, sources, infections
