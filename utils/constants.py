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

NEW_TOP_K = {3: [3, 6, 9, 12, 15],
             5: [5, 10, 15, 20, 25],
             10: [10, 20, 30, 40, 50],
             15: [15, 30, 45, 60, 75]}

MAKE_NEIGHBORS_POSITIVE = True

NAMES = {'ba': 'BA Network (1500 nodes) -', 'ba5000': 'BA Network (5000 nodes) -', 'er': 'ER Network (1500 nodes) -',
         'er5000': 'ER Network (5000 nodes) -', 'powergrid': 'Power Grid Network -',
         'facebookego': 'Facebook Ego Network -'}

MODEL_GRAPH_DIR = 'data/model/graph/'
MODEL_MIXED = 'data/model/mixed/'
MODEL_ABLATION = 'data/model/untransformed/'

COLORS = {'nb': {3: '#393b79', 5: '#5254a3', 10: '#6b6ecf', 15: '#9c9ede'},
          'og': {3: '#637939', 5: '#8ca252', 10: '#b5cf6b', 15: '#cedb9c'}}
