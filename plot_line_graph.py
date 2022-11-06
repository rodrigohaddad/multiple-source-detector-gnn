import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    aa = {'n_sources': [1, 1], 'top_k': [1, 2], 'precision_mean': [0.5, 0.5], 'recall_mean': [0.5, 0.6], 'infection_percentage': ['enriched', 'enriched'], 'f_score_mean': [0.5, 0.5333333373069763]}

    data_preproc = pd.DataFrame({
        'Year': [1, 2, 3, 4, 5],
        'A': [1, 1, 2, 4, 5],
        'B': [1, 2, 5, 4, 5],
        'C': [1, 2, 3, 4, 5],
        'D': [1, 2, 1, 4, 8]})

    axes = sns.lineplot(x='Year', y='value', hue='variable',
                        data=pd.melt(data_preproc, ['Year']))
    axes.set_title(f'Precision M:{data_preproc["A"].mean():.4f} V:{data_preproc["A"].var():.4f}')
    plt.savefig("data/figures/comparison/comparison_1", dpi=120)


if __name__ == '__main__':
    main()
