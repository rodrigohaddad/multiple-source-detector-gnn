import json
import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torchmetrics

from utils.constants import MODEL_GRAPH_DIR, NEW_TOP_K, MAKE_NEIGHBORS_POSITIVE
from utils.test_model import test_pred


def calculate_neighborhood_rate(data, indices_pred):
    sources = np.argwhere(data.y > 0)[0]
    neighbors_of_sources = np.array([])
    for source in sources:
        correct_pred = np.argwhere(indices_pred == int(source))[0]
        source_idx_0 = data.edge_index[1][np.argwhere(data.edge_index[0] == int(source))[0]]

        founds_0 = np.where(np.in1d(source_idx_0, indices_pred))[0]
        neighbors = source_idx_0[founds_0]
        neighbors_of_sources = np.append(neighbors_of_sources, neighbors)
        print("")
    uniq = np.unique(neighbors_of_sources)
    print("aa")


def make_neighborhood_positive(data, pred_index_sources):
    sources = np.argwhere(data.y > 0)[0]
    neighbors_of_sources = np.array([])
    for source in sources:
        nb_source_idx_0 = data.edge_index[1][np.argwhere(data.edge_index[0] == int(source))[0]]
        neighbors_of_sources = np.append(neighbors_of_sources, nb_source_idx_0)

    # May be discontinued
    # intersection = np.intersect1d(neighbors_of_sources, pred_index_sources)

    y = data.y.clone().detach()
    y[neighbors_of_sources] = 1
    return y


def test(neighbors_positive=MAKE_NEIGHBORS_POSITIVE):
    metrics_result = {'n_sources': [], 'top_k': [], 'infection_percentage': [], 'name': [],
                      'precision_mean': [], 'precision_var': [],
                      'recall_mean': [], 'recall_var': [],
                      'f_score_mean': [], 'f_score_var': []}
    # Load test graphs
    for path in os.listdir('data/graph_enriched/'):
        enriched_path = f'data/graph_enriched/{path}'
        sage_model_name = f'graph-sage-{enriched_path.split("/")[-1]}.pickle'
        sage = pickle.load(open(f'{MODEL_GRAPH_DIR}{sage_model_name}', 'rb'))
        sources = int(enriched_path.split('_')[3][:-1])
        infection = int(enriched_path.split('_')[2][:len(enriched_path.split('_')[2])-3])
        for top_k in NEW_TOP_K[sources]:
            data_plot = {'FPR': [], 'FNR': [], 'F-score': [], 'Precision': [], 'Recall': []}
            index = []
            fpr_arr = np.array([])
            fnr_arr = np.array([])
            f_score_arr = np.array([])
            precision_arr = np.array([])
            recall_arr = np.array([])

            for i, filename in enumerate(os.listdir(f'{enriched_path}/test')):  # /test
                if not os.path.isdir(f"data/figures/individual/{enriched_path.split('/')[-1]}"):
                    os.makedirs(f"data/figures/individual/{enriched_path.split('/')[-1]}")

                if not os.path.isdir(f"data/figures/individual/{enriched_path.split('/')[-1]}/nb"):
                    os.makedirs(f"data/figures/individual/{enriched_path.split('/')[-1]}/nb")

                graph_file = os.path.join(f'{enriched_path}/test', filename)  # /test
                if not os.path.isfile(graph_file):
                    continue

                # Data
                data = pickle.load(open(graph_file, 'rb'))
                # data.y = data.x[:, -1].float()
                data.y = data.x[:, -1].long()
                data.x = data.x[:, :-1].float()

                # Apply embedding model
                y_pred, indices = test_pred(sage, data, sources=top_k)

                y = data.y if not neighbors_positive else make_neighborhood_positive(data, indices)

                # Eval
                acc = torchmetrics.functional.accuracy(y_pred, y)
                f_score = torchmetrics.functional.f1_score(y_pred, y, task='binary')
                # sk_f = sk_f1_score(y, y_pred, average='micro')

                roc = torchmetrics.ROC()
                fpr, tpr, thresholds = roc(y_pred, y)
                fpr = fpr[1]
                tpr = tpr[1]

                fnr = 1 - tpr
                tnr = 1 - fpr

                prc = torchmetrics.functional.precision(y_pred, y)
                rec = torchmetrics.functional.recall(y_pred, y)

                # print(f'N sources pred: {sum(y_pred)}')
                # print(f'Acc: {acc}, F_score: {f_score}, {sk_f}')
                # print(f'FPR: {fpr}, FNR: {fnr}')
                # print(f'TPR: {tpr}, TNR: {tnr}')
                # print(f'Precision: {prc}')
                # print(f'Recall: {rec}')

                index.append(f'G_{i}')
                fpr_arr = np.append(fpr_arr, float(fpr))
                fnr_arr = np.append(fnr_arr, float(fnr))
                f_score_arr = np.append(f_score_arr, float(f_score))

                precision_arr = np.append(precision_arr, float(prc))
                recall_arr = np.append(recall_arr, float(rec))

                data_plot['FPR'].append(float(fpr))
                data_plot['FNR'].append(float(fnr))
                data_plot['F-score'].append(float(f_score))
                data_plot['Precision'].append(float(prc))
                data_plot['Recall'].append(float(rec))

            df = pd.DataFrame({'Precision': data_plot['Precision'],
                               'Recall': data_plot['Recall'],
                               'F-score': data_plot['F-score']},
                              index=index)

            axes = df.plot.bar(rot=0, subplots=True, grid=True,
                               color=['#FEBCC8', '#C8CFE7', '#C7E5C6'],
                               title=f'Test set - {sources} sources - Top {top_k}') # Test

            axes[1].legend(loc=2)
            axes[0].set_title(f'Precision M:{precision_arr.mean():.4f} V:{precision_arr.var():.4f}')
            axes[1].set_title(f'Recall M:{recall_arr.mean():.4f} V:{recall_arr.var():.4f}')
            axes[2].set_title(f'F-score M:{f_score_arr.mean():.4f} V:{f_score_arr.var():.4f}')

            plt.savefig(f"data/figures/individual/{enriched_path.split('/')[-1]}/{'nb/' if neighbors_positive else ''}{enriched_path.split('/')[-1]}-test-{top_k}topk",
                        dpi=120) #test

            metrics_result['name'].append(path.split('_')[0])
            metrics_result['n_sources'].append(sources)
            metrics_result['top_k'].append(top_k)
            metrics_result['infection_percentage'].append(infection)
            metrics_result['precision_mean'].append(precision_arr.mean())
            metrics_result['recall_mean'].append(recall_arr.mean())
            metrics_result['f_score_mean'].append(f_score_arr.mean())

            metrics_result['precision_var'].append(precision_arr.var())
            metrics_result['recall_var'].append(recall_arr.var())
            metrics_result['f_score_var'].append(f_score_arr.var())

    f = open(f'data/metrics_output/metrics_output{"_nb" if neighbors_positive else ""}.json', 'w')
    f.write(json.dumps(metrics_result))


if __name__ == '__main__':
    test(False)
