import json

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

METRICS_AGGREGATED_FILES = ['metrics_output_train.json', 'metrics_output_train_nb.json',
                            'metrics_output_test.json', 'metrics_output_test_nb.json']


def main():
    sns.set_style("whitegrid")
    for file_path in METRICS_AGGREGATED_FILES:
        f = open(f'data/{file_path}', 'r')
        data = pd.DataFrame(json.load(f))
        data_preproc_inf = data.where(data['infection_percentage'] == 5)
        data_preproc_src = data.where(data['n_sources'] == 3)

        for s, group in data_preproc_inf.groupby(['n_sources']):
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 5))
            sns.lineplot(x='top_k', y='precision_mean', data=group, ax=ax1, marker="o", linestyle='--', color='black')
            sns.lineplot(x='top_k', y='recall_mean', data=group, ax=ax2, marker="o", linestyle='--', color='black')
            sns.lineplot(x='top_k', y='f_score_mean', data=group, ax=ax3, marker="o", linestyle='--', color='black')
            ax1.set_title(f'Precision')
            ax2.set_title(f'Recall')
            ax3.set_title(f'F-score')
            fig.suptitle(
                f"{file_path.split('_')[2].split('.')[0]} - {int(s)} sources - 5% infection{' - Neighbors' if file_path.split('_')[-1].split('.')[0] == 'nb' else ''}")
            plt.savefig(f"data/figures/comparison/comparison_{int(s)}s_5inf_{file_path.split('_')[2].split('.')[0]}{'_nb' if file_path.split('_')[-1].split('.')[0] == 'nb' else ''}", dpi=120)

        for inf, group in data_preproc_src.groupby(['infection_percentage']):
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 5))
            sns.lineplot(x='top_k', y='precision_mean', data=group, ax=ax1, marker="o", linestyle='--', color='black')
            sns.lineplot(x='top_k', y='recall_mean', data=group, ax=ax2, marker="o", linestyle='--', color='black')
            sns.lineplot(x='top_k', y='f_score_mean', data=group, ax=ax3, marker="o", linestyle='--', color='black')
            ax1.set_title(f'Precision')
            ax2.set_title(f'Recall')
            ax3.set_title(f'F-score')
            fig.suptitle(f"{file_path.split('_')[2].split('.')[0]} - 3 sources - {int(inf)}% infection{' - Neighbors' if file_path.split('_')[-1].split('.')[0] == 'nb' else ''}")
            plt.savefig(f"data/figures/comparison/comparison_3s_{int(inf)}inf_{file_path.split('_')[2].split('.')[0]}{'_nb' if file_path.split('_')[-1].split('.')[0] == 'nb' else ''}", dpi=120)


if __name__ == '__main__':
    main()
