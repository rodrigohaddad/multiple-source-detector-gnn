import json

import pandas as pd


def sum_of_squares(x):
    return sum(x ** 2)


def main():
    df_base = pd.DataFrame(json.load(open(f'data/metrics_output/metrics_output_ablation_base.json', 'r')))
    df_enriched = pd.DataFrame(json.load(open(f'data/metrics_output/metrics_output_ablation.json', 'r')))

    result_base = df_base.groupby(['n_sources'])['f_score_mean'].agg(['max', 'mean', 'var'])
    var_base = df_base.groupby(['n_sources'])['f_score_var'].agg([sum_of_squares, 'count'])
    var_base["result"] = var_base["sum_of_squares"]/var_base["count"]
    result_base["var"] = result_base["var"] + var_base["result"]

    result_enriched = df_enriched.groupby(['n_sources'])['f_score_mean'].agg(['max', 'mean', 'var'])
    var_enriched = df_enriched.groupby(['n_sources'])['f_score_var'].agg([sum_of_squares, 'count'])
    var_enriched["result"] = var_enriched["sum_of_squares"] / var_enriched["count"]
    result_enriched["var"] = result_enriched["var"] + var_enriched["result"]

    print(result_base)
    # print(var_base)

    print(result_enriched)
    # print(var_enriched)


if __name__ == '__main__':
    main()
