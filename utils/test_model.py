import torch
import pickle

from utils.constants import INFECTED_DIR


@torch.no_grad()
def test_embedding(model, data, no_weight):
    model.eval()
    try:
        data.test_mask
    except:
        data.test_mask = torch.arange(data.num_nodes - 1)

    if no_weight:
        out = model.full_forward(data.x, data.edge_index).cpu()
    else:
        out = model.full_forward(data.x, data.edge_index, data.edge_weight).cpu()

    # _, out = model(data.x, data.edge_index, data.edge_weight)
    # acc = accuracy(out.argmax(dim=1)[data.test_mask], data.y[data.test_mask])
    return out


def concatenate_sources(file, filename, sources, conj_emb, emb=None):
    if emb is None:
        emb = pickle.load(open(file, 'rb'))
    inf_model = pickle.load(open(f'{INFECTED_DIR}/{filename.split("-")[0]}-infected.pickle', 'rb'))

    infections = list(inf_model.model.status.values())
    emb = torch.column_stack((emb, torch.Tensor(infections)))

    sources = torch.concat((sources, torch.IntTensor(list(inf_model.model.initial_status.values()))))
    conj_emb = torch.concat((conj_emb, emb))

    return conj_emb, sources, infections
