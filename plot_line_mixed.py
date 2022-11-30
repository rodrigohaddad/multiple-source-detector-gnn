import json

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

METRICS_AGGREGATED_TRAIN = {'train': ['metrics_output_mixed_train.json', 'metrics_output_mixed_train_nb.json'],
                            'test': ['metrics_output_mixed_test.json', 'metrics_output_mixed_test_nb.json']}

METRICS_AGGREGATED_OG = {'og': ['metrics_output_mixed_train.json', 'metrics_output_mixed_test.json'],
                         'nb': ['metrics_output_mixed_train_nb.json', 'metrics_output_mixed_test_nb.json']}


def main():
    sns.set_style("whitegrid")
    for key, (file_path, file_path_nb) in METRICS_AGGREGATED_OG.items():
        f_train = open(f'data/{file_path}', 'r')
        data_train = pd.DataFrame(json.load(f_train))

        f_test = open(f'data/{file_path_nb}', 'r')
        data_test = pd.DataFrame(json.load(f_test))


        for (s, group_train), (s_test, group_test) in zip(data_train.groupby(['n_sources']),
                                                          data_test.groupby(['n_sources'])):
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 5))
            sns.lineplot(x='top_k', y='precision_mean', data=group_train, ax=ax1, marker="o", linestyle='--',
                         color='#009337')
            sns.lineplot(x='top_k', y='recall_mean', data=group_train, ax=ax2, marker="o", linestyle='--',
                         color='#009337')
            sns.lineplot(x='top_k', y='f_score_mean', data=group_train, ax=ax3, marker="o", linestyle='--',
                         color='#009337')

            sns.lineplot(x='top_k', y='precision_mean', data=group_test, ax=ax1, marker="o", linestyle='--',
                         color='#840000')
            sns.lineplot(x='top_k', y='recall_mean', data=group_test, ax=ax2, marker="o", linestyle='--',
                         color='#840000')
            sns.lineplot(x='top_k', y='f_score_mean', data=group_test, ax=ax3, marker="o", linestyle='--',
                         color='#840000')
            ax1.legend(['Train', 'Test'])
            ax2.legend(['Train', 'Test'])
            ax3.legend(['Train', 'Test'])

            ax1.set_title(f'Precision')
            ax2.set_title(f'Recall')
            ax3.set_title(f'F-score')
            fig.suptitle(
                f"{int(s)} source{'' if s == 1 else 's'} - 15% infection{' - Neighbors' if key == 'nb' else ''}")
            plt.savefig(f"data/figures/comparison/comparison_mixed_train_test/comparison_{int(s)}s_15inf{'_nb' if key == 'nb' else ''}", dpi=120)

    for key, (file_path, file_path_nb) in METRICS_AGGREGATED_TRAIN.items():
        f_train = open(f'data/{file_path}', 'r')
        data_train = pd.DataFrame(json.load(f_train))

        f_test = open(f'data/{file_path_nb}', 'r')
        data_test = pd.DataFrame(json.load(f_test))

        for (s, group_train), (s_test, group_test) in zip(data_train.groupby(['n_sources']),
                                                          data_test.groupby(['n_sources'])):
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 5))
            sns.lineplot(x='top_k', y='precision_mean', data=group_train, ax=ax1, marker="o", linestyle='--',
                         color='#a87900')
            sns.lineplot(x='top_k', y='recall_mean', data=group_train, ax=ax2, marker="o", linestyle='--',
                         color='#a87900')
            sns.lineplot(x='top_k', y='f_score_mean', data=group_train, ax=ax3, marker="o", linestyle='--',
                         color='#a87900')

            sns.lineplot(x='top_k', y='precision_mean', data=group_test, ax=ax1, marker="o", linestyle='--',
                         color='#2976bb')
            sns.lineplot(x='top_k', y='recall_mean', data=group_test, ax=ax2, marker="o", linestyle='--',
                         color='#2976bb')
            sns.lineplot(x='top_k', y='f_score_mean', data=group_test, ax=ax3, marker="o", linestyle='--',
                         color='#2976bb')
            ax1.legend(['Original', 'Neighborhood'])
            ax2.legend(['Original', 'Neighborhood'])
            ax3.legend(['Original', 'Neighborhood'])

            ax1.set_title(f'Precision')
            ax2.set_title(f'Recall')
            ax3.set_title(f'F-score')
            fig.suptitle(
                f"{int(s)} source{'' if s == 1 else 's'} - 15% infection - {key.capitalize()}")
            plt.savefig(f"data/figures/comparison/comparison_mixed_og_nb/comparison_{int(s)}s_15inf_{key}", dpi=120)


if __name__ == '__main__':
    main()
