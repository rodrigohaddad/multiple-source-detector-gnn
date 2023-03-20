import json
import os
import pickle
from statistics import mean

import networkx as nx
import pandas as pd

from utils.constants import INFECTED_DIR

NAMES = ['ba_']


# def main():
#     f = open(f'data/metrics_output/basic_metrics.json', 'w')
#     g_dir = os.listdir(INFECTED_DIR)
#     dict_results = {'diameter': [], 'nodes': [],
#                     'edges': [], 'avg_degree': []}
#     name_results = dict()
#     for name in NAMES:
#         name_results[name] = dict_results
#         g_dirs_filtered = [k for k in g_dir if name in k]
#         for g_dir_filtered in g_dirs_filtered:
#             path = os.path.join(INFECTED_DIR, g_dir_filtered)
#             files = os.listdir(path)
#             for file_name in files:
#                 file = os.path.join(path, file_name)
#                 g_inf = pickle.load(open(file, 'rb')).G
#                 diameter = nx.diameter(g_inf)
#                 print(f'{name} diameter: {diameter}')
#                 name_results[name]['diameter'].append(diameter)
#
#                 nodes = len(g_inf)
#                 print(f'{name} nodes: {nodes}')
#                 name_results[name]['nodes'].append(nodes)
#
#                 edges = g_inf.number_of_edges()
#                 print(f'{name} edges: {edges}')
#                 name_results[name]['edges'].append(edges)
#
#                 avg_degree = 2*g_inf.number_of_edges() / len(g_inf)
#                 print(f'{name} avg_degree: {avg_degree}')
#                 name_results[name]['avg_degree'].append(avg_degree)
#
#         print(f'{name} diameter: {mean(name_results[name]["diameter"])}')
#         print(f'{name} nodes: {mean(name_results[name]["nodes"])}')
#         print(f'{name} edges: {mean(name_results[name]["edges"])}')
#         print(f'{name} avg_degree: {mean(name_results[name]["avg_degree"])}')
#
#     f.write(json.dumps(name_results))

def main():
    df = pd.DataFrame(json.load(open(f'data/metrics_output/metrics_output.json', 'r')))

    df = df[(df['infection_percentage'] == 30) & (df['name'].isin(['powergrid', 'facebookego']))]

    result = df.groupby(['n_sources', 'name'])['f_score_mean'].agg(['mean', 'max'])

    print(result)



if __name__ == '__main__':
    main()
