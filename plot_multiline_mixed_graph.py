import json

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

F = {'og': open(f'data/metrics_output_test.json', 'r'),
     'nb': open(f'data/metrics_output_test_nb.json', 'r')}

SOURCES = [3, 5, 10, 15]
INFECTIONS = [5, 10, 15, 20, 30]


def main():
    sns.set_style("whitegrid")

    for inf in INFECTIONS:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 5))

        for key, content in F.items():
            data = pd.DataFrame(json.load(content))
            data_inf = data.where(data['infection_percentage'] == inf)
            for s in SOURCES:
                loc = 'upper right' if key == 'nb' else 'lower right'

                data_inf_s = data_inf.where(data['n_sources'] == s)
                line1, = sns.lineplot(x='top_k', y='precision_mean', data=data_inf_s, ax=ax1, marker="o",
                                      linestyle='--',
                                      color='#840000', label=f'{s}')
                line2, = sns.lineplot(x='top_k', y='recall_mean', data=data_inf_s, ax=ax2, marker="o", linestyle='--',
                                      color='#840000', label=f'{s}')
                line3, = sns.lineplot(x='top_k', y='f_score_mean', data=data_inf_s, ax=ax3, marker="o", linestyle='--',
                                      color='#840000', label=f'{s}')

                leg1 = ax1.legend(handles=[line1], loc=loc)
                ax1.add_artist(leg1)
                leg2 = ax2.legend(handles=[line2], loc=loc)
                ax2.add_artist(leg2)
                leg3 = ax3.legend(handles=[line3], loc=loc)
                ax3.add_artist(leg3)

        plt.legend()

        ax1.set_title(f'Precision')
        ax2.set_title(f'Recall')
        ax3.set_title(f'F-score')

        fig.suptitle(
            f"{inf}% infection")
        plt.savefig(
            f"data/figures/comparison/comparison_infection_multiline/comparison_{inf}inf",
            dpi=120)


if __name__ == '__main__':
    main()
