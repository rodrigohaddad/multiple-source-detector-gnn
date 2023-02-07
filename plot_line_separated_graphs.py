import json

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils.constants import COLORS

F = {'og': pd.DataFrame(json.load(open(f'data/metrics_output/metrics_output_mixed.json', 'r'))),
     'nb': pd.DataFrame(json.load(open(f'data/metrics_output/metrics_output_mixed_nb.json', 'r')))}

SOURCES = [3, 5, 10, 15]
SYMBOLS = ['o', '^', 's', '*']
INFECTIONS = [5, 10, 20, 30]

NAMES = {'ba': 'BA Network (1500 nodes) -', 'ba5000': 'BA Network (5000 nodes) -', 'er': 'ER Network -',
         'powergrid': 'Power Grid Network -', 'facebookego': 'Facebook Ego Network -'}

graph_types = {'Precision': ['precision mean', 'precision_mean'],
               'Recall': ['recall mean', 'recall_mean'],
               'F-score': ['f-score mean', 'f_score_mean']}


def main():
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 13})
    my_xticks = ['s', '2s', '3s', '4s', '5s']

    og_names = F['og'].groupby(['name'])
    nb_names = F['nb'].groupby(['name'])

    for og_name, nb_name in zip(og_names, nb_names):
        for inf in INFECTIONS:
            for graph_type, (y_name, column_name) in graph_types.items():
                fig, ax1 = plt.subplots(figsize=(7, 5),
                                        layout='constrained'
                                        )
                sources_lines_og_1 = []
                sources_lines_nb_1 = []
                for key, (name, data) in zip(['og', 'nb'], [og_name, nb_name]):
                    data_inf = data[data['infection_percentage'] == inf]
                    name = name
                    for s, symbol in zip(SOURCES, SYMBOLS):
                        data_inf_s = data_inf[data_inf['n_sources'] == s]
                        data_inf_s = data_inf_s.sort_values(by=['top_k'])
                        data_inf_s['top_k_literal'] = my_xticks
                        line1 = ax1.plot('top_k_literal', column_name, data=data_inf_s,
                                         marker=symbol,
                                         linestyle='--',
                                         color=COLORS[key][s],
                                         label=f'{s}')
                        if key == 'nb':
                            sources_lines_nb_1.append(line1[0])
                        else:
                            sources_lines_og_1.append(line1[0])

                first_legend_1 = ax1.legend(handles=sources_lines_nb_1, loc='lower left', bbox_to_anchor=(1, 0.3),
                                            title='w neighbors')

                ax1.add_artist(first_legend_1)
                ax1.legend(handles=sources_lines_og_1, loc='lower left', bbox_to_anchor=(1, 0.65),
                           title='w/o neighbors')

                ax1.set_title(graph_type)
                ax1.set_xlabel('top k')
                ax1.set_ylabel(y_name)

                fig.suptitle(f"{NAMES[name]} {inf}% Infection", fontweight='bold')
                plt.savefig(
                    f"data/figures/comparison/comparison_infection_multiline_separated/comparison_{name}_{column_name}_{inf}inf",
                    dpi=120)


if __name__ == '__main__':
    main()
