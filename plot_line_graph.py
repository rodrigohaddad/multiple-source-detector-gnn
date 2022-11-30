import json

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

METRICS_AGGREGATED_FILES = ['metrics_output_train.json', 'metrics_output_train_nb.json',
                            'metrics_output_test.json', 'metrics_output_test_nb.json']

METRICS_AGGREGATED = {'og': ['metrics_output_train.json', 'metrics_output_test.json'],
                      'nb': ['metrics_output_train_nb.json', 'metrics_output_test_nb.json']}


def main():
    sns.set_style("whitegrid")
    for key, (file_path_train, file_path_test) in METRICS_AGGREGATED.items():
        f_train = open(f'data/{file_path_train}', 'r')
        data_train = pd.DataFrame(json.load(f_train))
        data_preproc_inf_train = data_train.where(data_train['infection_percentage'] == 5)
        data_preproc_src_train = data_train.where(data_train['n_sources'] == 3)
        data_preproc_inf_many_src_train = data_train.where((data_train['infection_percentage'] == 15) &
                                                           (data_train['n_sources'] != 3))

        f_test = open(f'data/{file_path_test}', 'r')
        data_test = pd.DataFrame(json.load(f_test))
        data_preproc_inf_test = data_test.where(data_test['infection_percentage'] == 5)
        data_preproc_src_test = data_test.where(data_test['n_sources'] == 3)
        data_preproc_inf_many_src_test = data_test.where((data_test['infection_percentage'] == 15) &
                                                          (data_test['n_sources'] != 3))

        for (s, group_train), (s_test, group_test) in zip(data_preproc_inf_train.groupby(['n_sources']),
                                                          data_preproc_inf_test.groupby(['n_sources'])):
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
                f"{int(s)} source{'' if s == 1 else 's'} - 5% infection{' - Neighbors' if key == 'nb' else ''}")
            plt.savefig(f"data/figures/comparison_train_test/comparison_{int(s)}s_5inf{'_nb' if key == 'nb' else ''}", dpi=120)

        for (inf, group_train), (inf_test, group_test) in zip(data_preproc_src_train.groupby(['infection_percentage']),
                                                              data_preproc_src_test.groupby(['infection_percentage'])):
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
            fig.suptitle(f"3 sources - {int(inf)}% infection{' - Neighbors' if key == 'nb' else ''}")
            plt.savefig(f"data/figures/comparison_train_test/comparison_3s_{int(inf)}inf{'_nb' if key == 'nb' else ''}", dpi=120)

        for (s, group_train), (s_test, group_test) in zip(data_preproc_inf_many_src_train.groupby(['n_sources']),
                                                          data_preproc_inf_many_src_test.groupby(['n_sources'])):
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
            plt.savefig(f"data/figures/comparison_train_test/comparison_{int(s)}s_15inf{'_nb' if key == 'nb' else ''}", dpi=120)


if __name__ == '__main__':
    main()
