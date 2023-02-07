import numpy as np


def calculate_avg_degree(G):
    degrees = np.array([int(dg) for (node, dg) in G.degree()])
    print(f"Mean: {degrees.mean()}")
