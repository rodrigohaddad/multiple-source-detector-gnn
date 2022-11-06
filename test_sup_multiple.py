import json
import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from torch_geometric.utils import accuracy, f1_score, false_positive, true_positive, precision, recall
from sklearn.metrics import f1_score as sk_f1_score

from utils.constants import MODEL_GRAPH_DIR, GRAPH_ENRICHED, TOP_K, MAKE_NEIGHBORS_POSITIVE
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
    intersection = np.intersect1d(neighbors_of_sources, pred_index_sources)

    y = data.y.clone().detach()
    y[intersection] = 1
    return y


def main():
    metrics_result = {'n_sources': [], 'top_k': [],
                      'precision_mean': [], 'recall_mean': [],
                      'infection_percentage': [], 'f_score_mean': []}
    # Load test graphs
    for enriched_path in GRAPH_ENRICHED:
        sage_model_name = f'graph-sage-{enriched_path.split("/")[-1]}.pickle'
        sage = pickle.load(open(f'{MODEL_GRAPH_DIR}{sage_model_name}', 'rb'))
        sources = int(enriched_path[-2])
        infection = '_'.split(enriched_path)[1][-4:]
        for top_k in TOP_K[sources]:
            data_plot = {'FPR': [], 'FNR': [], 'F-score': [], 'Precision': [], 'Recall': []}
            index = []
            fpr_arr = np.array([])
            fnr_arr = np.array([])
            f_score_arr = np.array([])
            precision_arr = np.array([])
            recall_arr = np.array([])

            for i, filename in enumerate(os.listdir(f'{enriched_path}')):  # /test
                graph_file = os.path.join(f'{enriched_path}', filename)  # /test
                if not os.path.isfile(graph_file):
                    continue

                # Data
                data = pickle.load(open(graph_file, 'rb'))
                # data.y = data.x[:, -1].float()
                data.y = data.x[:, -1].long()
                data.x = data.x[:, :-1].float()

                # Apply embedding model
                y_pred, indices = test_pred(sage, data, sources=top_k)

                y = data.y if not MAKE_NEIGHBORS_POSITIVE else make_neighborhood_positive(data, indices)

                # Eval
                acc = accuracy(y, y_pred)
                f_score = f1_score(y_pred, y, 2)
                sk_f = sk_f1_score(y, y_pred, average='micro')

                fn, fp = false_positive(y_pred, y, 2)
                tn, tp = true_positive(y_pred, y, 2)

                prc = precision(y_pred, y, 2)
                rec = recall(y_pred, y, 2)

                print(f'N sources pred: {sum(y_pred)}')
                print(f'Acc: {acc}, F_score: {f_score}, {sk_f}')
                print(f'FPR: {fp / (tn + fp)}, FNR: {fn / (tp + fn)}')
                print(f'TPR: {tp / (tp + fn)}, TNR: {tn / (tn + fp)}')
                print(f'Precision: {prc}')
                print(f'Recall: {rec}')

                index.append(f'G_{i}')
                fpr_arr = np.append(fpr_arr, float(fp / (tn + fp)))
                fnr_arr = np.append(fnr_arr, float(fn / (tp + fn)))
                f_score_arr = np.append(f_score_arr, float(f_score[1]))

                precision_arr = np.append(precision_arr, float(prc[1]))
                recall_arr = np.append(recall_arr, float(rec[1]))

                data_plot['FPR'].append(float(fp / (tn + fp)))
                data_plot['FNR'].append(float(fn / (tp + fn)))
                data_plot['F-score'].append(float(f_score[1]))
                data_plot['Precision'].append(float(prc[1]))
                data_plot['Recall'].append(float(rec[1]))

            df = pd.DataFrame({'Precision': data_plot['Precision'],
                               'Recall': data_plot['Recall'],
                               'F-score': data_plot['F-score']},
                              index=index)

            axes = df.plot.bar(rot=0, subplots=True, grid=True,
                               color=['#FEBCC8', '#C8CFE7', '#C7E5C6'],
                               title=f'Train set - {sources} sources - Top {top_k}')
            axes[1].legend(loc=2)
            axes[0].set_title(f'Precision M:{precision_arr.mean():.4f} V:{precision_arr.var():.4f}')
            axes[1].set_title(f'Recall M:{recall_arr.mean():.4f} V:{recall_arr.var():.4f}')
            axes[2].set_title(f'F-score M:{f_score_arr.mean():.4f} V:{f_score_arr.var():.4f}')
            plt.savefig(f"data/figures/{enriched_path.split('/')[-1]}/{'nb/' if MAKE_NEIGHBORS_POSITIVE else ''}{enriched_path.split('/')[-1]}-train-{top_k}topk",
                        dpi=120)

            metrics_result['n_sources'].append(sources)
            metrics_result['top_k'].append(top_k)
            metrics_result['infection_percentage'].append(infection)
            metrics_result['precision_mean'].append(precision_arr.mean())
            metrics_result['recall_mean'].append(recall_arr.mean())
            metrics_result['f_score_mean'].append(f_score_arr.mean())

    f = open(f'metrics_output_train{"_nb" if MAKE_NEIGHBORS_POSITIVE else ""}.json', 'w')
    f.write(json.dumps(metrics_result))


if __name__ == '__main__':
    main()
