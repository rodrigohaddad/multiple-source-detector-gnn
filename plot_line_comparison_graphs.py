import json

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils.constants import COLORS

F = {'og': pd.DataFrame(json.load(open(f'data/metrics_output/metrics_output_mixed.json', 'r'))),
     'nb': pd.DataFrame(json.load(open(f'data/metrics_output/metrics_output_mixed_nb.json', 'r')))}

SOURCES = [5, 10]
SYMBOLS = ['o', '^']
INFECTIONS = [10, 20, 30]

NAMES = {'ba': 'BA Network (1500 nodes)', 'ba5000': 'BA Network (5000 nodes)', 'er': 'ER Network (1500 nodes)',
         'er5000': 'ER Network (5000 nodes)', 'powergrid': 'Power Grid Network',
         'facebookego': 'Facebook Ego Network'}

graph_types = {
    # 'Precision': ['precision mean', 'precision_mean'],
    # 'Recall': ['recall mean', 'recall_mean'],
    'F-score': ['f-score mean', 'f_score_mean']
}

color_line = {10: '#009337', 20: '#014182', 30: '#9a0200'}


def main(source_type='nb'):
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 13})
    my_xticks = ['s', '2s', '3s', '4s', '5s']

    grouped = F[source_type].groupby(['name'])

    for name, data in grouped:
        fig, ax1 = plt.subplots(figsize=(7, 5),
                                layout='constrained')
        for graph_type, (y_name, column_name) in graph_types.items():
            sources_lines = {10: [], 20: [], 30: []}
            for inf in INFECTIONS:
                data_inf = data[data['infection_percentage'] == inf]
                for s, symbol in zip(SOURCES, SYMBOLS):
                    # if graph_type == "F-score" and name == "ba":
                    #     print()
                    data_inf_s = data_inf[data_inf['n_sources'] == s]
                    data_inf_s = data_inf_s.sort_values(by=['top_k'])
                    data_inf_s['top_k_literal'] = my_xticks
                    line1 = ax1.plot('top_k_literal', column_name, data=data_inf_s,
                                     marker=symbol,
                                     linestyle='--',
                                     color=color_line[inf],
                                     label=f'{s} sources')

                    sources_lines[inf].append(line1[0])

            first_legend_1 = ax1.legend(handles=sources_lines[10], loc='lower left',
                                        bbox_to_anchor=(1, 0.75), title='10% infected')

            ax1.add_artist(first_legend_1)
            second_legend = ax1.legend(handles=sources_lines[20], loc='lower left', bbox_to_anchor=(1, 0.45),
                                       title='20% infected')

            ax1.add_artist(second_legend)
            ax1.legend(handles=sources_lines[30], loc='lower left', bbox_to_anchor=(1, 0.15),
                       title='30% infected')

            ax1.set_title(graph_type)
            ax1.set_xlabel('top k')
            ax1.set_ylabel(y_name)

            fig.suptitle(f"{NAMES[name]}", fontweight='bold')
            plt.savefig(
                f"data/figures/comparison/comparison_infection_two_line/comparison_{name}_{column_name}_{source_type}",
                dpi=120)


if __name__ == '__main__':
    for i in ['nb', 'og']:
        main(i)
