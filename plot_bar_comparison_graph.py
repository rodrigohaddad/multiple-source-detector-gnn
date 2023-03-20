import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils.constants import NAMES

f_og = {'individual': pd.DataFrame(json.load(open(f'data/metrics_output/metrics_output.json', 'r'))),
        'mixed': pd.DataFrame(json.load(open(f'data/metrics_output/metrics_output_mixed.json', 'r')))}

f_nb = {'individual': pd.DataFrame(json.load(open(f'data/metrics_output/metrics_output_nb.json', 'r'))),
        'mixed': pd.DataFrame(json.load(open(f'data/metrics_output/metrics_output_mixed_nb.json', 'r')))}

Y_LIM = {'nb': {'ba': 0.2, 'ba5000': 0.1, 'er': 0.25, 'er5000': 0.13, 'facebookego': 0.15, 'powergrid': 0.35},
         'og': {'ba': 0.21, 'ba5000': 0.14, 'er': 0.25, 'er5000': 0.18, 'facebookego': 0.28, 'powergrid': 0.23}}


def main():
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 18})
    for f, typ in zip([f_og, f_nb], ['og', 'nb']):
        df_grouped_individual = f['individual'].groupby(['name', 'infection_percentage', 'n_sources']).apply(
            lambda x: x.sort_values('f_score_mean', ascending=False).iloc[0])
        df_grouped_individual.reset_index(drop=True, inplace=True)
        grouped_individual = df_grouped_individual.groupby(['name', 'infection_percentage'])

        df_grouped_mixed = f['mixed'].groupby(['name', 'infection_percentage', 'n_sources']).apply(
            lambda x: x.sort_values('f_score_mean', ascending=False).iloc[0])
        df_grouped_mixed.reset_index(drop=True, inplace=True)
        grouped_mixed = df_grouped_mixed.groupby(['name', 'infection_percentage'])

        for (info_ind, group_ind), (info_mix, group_mix) in zip(grouped_individual, grouped_mixed):
            bar_width = 0.4
            index = np.arange(4)

            fig, ax = plt.subplots()
            fig.set_tight_layout(True)
            bar_positions = range(len(group_ind['n_sources']))
            ax.bar(bar_positions, group_ind['f_score_mean'],
                   bar_width, label='Individual', color=(0.0, 0.2, 0.5, 0.7))
            ax.bar([p + bar_width for p in bar_positions], group_mix['f_score_mean'],
                   bar_width, label='General', color=(0.6, 0.1, 0.3, 0.7))
            ax.set_ylim([0, Y_LIM[typ][info_ind[0]]])
            ax.set_xlabel('Number of sources')
            ax.set_ylabel('F-score')
            ax.set_xticks(index + bar_width / 2)
            ax.set_xticklabels(["3", "5", "10", "15"])
            ax.legend()

            fig.suptitle(
                f"{NAMES[info_ind[0]]} {info_ind[1]}% Infection", fontweight='bold')
            plt.savefig(
                f"data/figures/comparison/comparison_bar_{typ}/comparison_{info_ind[0]}_{info_ind[1]}inf",
                dpi=120)


if __name__ == '__main__':
    main()
