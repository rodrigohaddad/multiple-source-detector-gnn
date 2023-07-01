import json

import pandas as pd

NAMES = ['ba_']


def main():
    df = pd.DataFrame(json.load(open(f'data/metrics_output/metrics_output.json', 'r')))
    df = df[(df['infection_percentage'] == 30) & (df['name'].isin(['powergrid', 'facebookego']))]
    result = df.groupby(['n_sources', 'name'])['f_score_mean'].agg(['max', 'mean'])

    print(result)


if __name__ == '__main__':
    main()
