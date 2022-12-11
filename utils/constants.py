import torch

DEVICE = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

INFECTED_DIR = 'data/infected_graph'

NOT_TRANSFORMED_DIR = 'data/graph_enriched'

VAL_TRANSFORMED_DIR = 'data/graph_enriched/val'

TOP_K = {1: [1, 2, 5, 7, 10],
         2: [2, 4, 8, 16, 20],
         3: [3, 6, 12, 24, 30],
         5: [5, 10, 20, 40, 50],
         7: [7, 14, 28, 56, 70],
         10: [10, 20, 40, 80, 100],
         15: [15, 30, 60, 120, 150],
         20: [20, 40, 80, 160, 200]}

MAKE_NEIGHBORS_POSITIVE = False

MODEL_GRAPH_DIR = 'data/model/graph/'
MODEL_MIXED = 'data/model/mixed/'

COLORS = {'nb': {3: '#393b79', 5: '#5254a3', 10: '#6b6ecf', 15: '#9c9ede'},
          'og': {3: '#637939', 5: '#8ca252', 10: '#b5cf6b', 15: '#cedb9c'}}
