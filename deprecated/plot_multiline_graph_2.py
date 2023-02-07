import json

import matplotlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils.constants import COLORS

F = {'og': pd.DataFrame(json.load(open(f'data/metrics_output/metrics_output.json', 'r'))),
     'nb': pd.DataFrame(json.load(open(f'data/metrics_output/metrics_output_nb.json', 'r')))}

SOURCES = [3, 5, 10, 15]
SYMBOLS = ['o', '^', 's', '*']
INFECTIONS = [5, 10, 20, 30]

NAMES = {'ba': 'BA Network -', 'er': 'ER Network -',
         'powergrid': 'Power Grid Network -', 'facebookego': 'Facebook Ego Network -'}


def main():
    sns.set_style("whitegrid")
    my_xticks = ['s', '2s', '3s', '4s', '5s']
    # cmap = plt.get_cmap('tab20b')
    # for c in cmap.colors:
    #     print(matplotlib.colors.to_hex(c))

    og_names = F['og'].groupby(['name'])
    nb_names = F['nb'].groupby(['name'])

    for og_name, nb_name in zip(og_names, nb_names):
        for inf in INFECTIONS:
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            plt.subplots_adjust(hspace=0.25)
            ax1, ax2, ax3, ax4 = axes.flat
            fig.delaxes(ax4)
            sources_lines_og_1 = []
            sources_lines_og_2 = []
            sources_lines_og_3 = []
            sources_lines_nb_1 = []
            sources_lines_nb_2 = []
            sources_lines_nb_3 = []
            for key, (name, data) in zip(['og', 'nb'], [og_name, nb_name]):
                data_inf = data[data['infection_percentage'] == inf]
                name = name
                for s, symbol in zip(SOURCES, SYMBOLS):
                    # loc = 'upper right' if key == 'nb' else 'lower right'
                    data_inf_s = data_inf[data_inf['n_sources'] == s]
                    data_inf_s = data_inf_s.sort_values(by=['top_k'])
                    data_inf_s['top_k_literal'] = my_xticks
                    line1 = ax1.plot('top_k_literal', 'precision_mean', data=data_inf_s,
                                     marker=symbol,
                                     linestyle='--',
                                     color=COLORS[key][s],
                                     label=f'{s}')
                    line2 = ax2.plot('top_k_literal', 'recall_mean', data=data_inf_s,
                                     marker=symbol,
                                     linestyle='--',
                                     color=COLORS[key][s],
                                     label=f'{s}')
                    line3 = ax3.plot('top_k_literal', 'f_score_mean', data=data_inf_s,
                                     marker=symbol,
                                     linestyle='--',
                                     color=COLORS[key][s],
                                     label=f'{s}')
                    if key == 'nb':
                        sources_lines_nb_1.append(line1[0])
                        sources_lines_nb_2.append(line2[0])
                        sources_lines_nb_3.append(line3[0])
                    else:
                        sources_lines_og_1.append(line1[0])
                        sources_lines_og_2.append(line2[0])
                        sources_lines_og_3.append(line3[0])

            first_legend_1 = ax3.legend(handles=sources_lines_nb_1, loc='lower left', bbox_to_anchor=(1.15, 0.5),
                                        title='w neighbors')

            # first_legend_1 = ax1.legend(handles=sources_lines_nb_1, loc='lower left', mode='expand', ncol=3,
            #                             bbox_to_anchor=(0, 1.02, 1, 0.2),
            #                             title='w neighbors')

            ax3.add_artist(first_legend_1)
            ax3.legend(handles=sources_lines_og_1, loc='lower left', bbox_to_anchor=(1.15, 0.1), title='w/o neighbors')

            # first_legend_2 = ax2.legend(handles=sources_lines_nb_2, loc='upper left', bbox_to_anchor=(1, 0.5),
            #                             title='w neighbors')
            # ax2.add_artist(first_legend_2)
            # ax2.legend(handles=sources_lines_og_2, loc='upper left', bbox_to_anchor=(1, 1), title='w/o neighbors')
            #
            # first_legend_3 = ax3.legend(handles=sources_lines_nb_3, loc='upper left', bbox_to_anchor=(1, 0.5),
            #                             title='w neighbors')
            # ax3.add_artist(first_legend_3)
            # ax3.legend(handles=sources_lines_og_3, loc='upper left', bbox_to_anchor=(1, 1), title='w/o neighbors')

            ax1.set_title(f'Precision')
            ax1.set_xlabel('top k')
            ax1.set_ylabel('precision mean')

            ax2.set_title(f'Recall')
            ax2.set_xlabel('top k')
            ax2.set_ylabel('recall mean')

            ax3.set_title(f'F-score')
            ax3.set_xlabel('top k')
            ax3.set_ylabel('f-score mean')

            fig.suptitle(
                f"{NAMES[name]} {inf}% Infection", fontweight='bold')
            plt.savefig(
                f"data/figures/comparison/comparison_infection_multiline_2/comparison_{name}_{inf}inf",
                dpi=120, bbox_inches='tight')


if __name__ == '__main__':
    main()
