import copy

import networkx as nx
import numpy as np


def calculate_propagation_score(g_inf, alpha=.6):
    G = g_inf.G
    W = nx.to_numpy_array(G)
    D = np.diag(W.sum(axis=0))
    D_inv_sqrt = np.sqrt(np.linalg.inv(D))
    S_first = np.dot(D_inv_sqrt, W)
    S = np.dot(S_first, D_inv_sqrt)

    Y = g_inf.model.status
    Gt = copy.deepcopy(Y)
    for _ in range(30):
        Gt_old = copy.deepcopy(Gt)
        for node, inf in Gt.items():
            accum = 0
            for j in G.neighbors(node):
                accum = accum + S[node][j]*Gt_old[j]
            Gt[node] = alpha*accum+(1-alpha)*Y[node]

    return Gt
