import json
import os
import pickle
import numpy as np
import torchmetrics

from utils.constants import MAKE_NEIGHBORS_POSITIVE, MODEL_MIXED, NOT_TRANSFORMED_DIR, NEW_TOP_K
from utils.test_model import test_pred


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


def test_mixed(neighbors_positive=MAKE_NEIGHBORS_POSITIVE):
    metrics_result = {'n_sources': [], 'top_k': [], 'infection_percentage': [], 'name': [],
                      'precision_mean': [], 'precision_var': [],
                      'recall_mean': [], 'recall_var': [],
                      'f_score_mean': [], 'f_score_var': []}
    # Load test graphs
    for sage_model in os.listdir(MODEL_MIXED):
        sage = pickle.load(open(os.path.join(MODEL_MIXED, sage_model), 'rb'))
        all_dir_names = os.listdir(NOT_TRANSFORMED_DIR)
        dir_names = [s for s in all_dir_names if f"{sage_model.split('-')[2][:-7]}_" in s]

        for direct in dir_names:
            for top_k in NEW_TOP_K[int(direct.split('_')[2][:-1])]:
                data_plot = {'FPR': [], 'FNR': [], 'F-score': [], 'Precision': [], 'Recall': []}
                index = []
                fpr_arr = np.array([])
                fnr_arr = np.array([])
                f_score_arr = np.array([])
                precision_arr = np.array([])
                recall_arr = np.array([])

                for i, filename in enumerate(os.listdir(os.path.join(NOT_TRANSFORMED_DIR, f'{direct}/test'))):  # /test
                    file_name = os.path.join(NOT_TRANSFORMED_DIR, f'{direct}/test', filename)  # /test
                    if not os.path.isfile(file_name):
                        continue

                    # Data
                    data = pickle.load(open(file_name, 'rb'))
                    # data.y = data.x[:, -1].float()
                    data.y = data.x[:, -1].long()
                    data.x = data.x[:, :-1].float()

                    # Apply embedding model
                    y_pred, indices = test_pred(sage, data, sources=top_k)

                    y = data.y if not neighbors_positive else make_neighborhood_positive(data, indices)

                    # Eval
                    f_score = torchmetrics.functional.f1_score(y_pred, y, task='binary')

                    roc = torchmetrics.ROC()
                    fpr, tpr, thresholds = roc(y_pred, y)
                    fpr = fpr[1]
                    tpr = tpr[1]

                    fnr = 1 - tpr
                    tnr = 1 - fpr

                    prc = torchmetrics.functional.precision(y_pred, y)
                    rec = torchmetrics.functional.recall(y_pred, y)

                    # print(f'N sources pred: {sum(y_pred)}')
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

                metrics_result['name'].append(sage_model.split('-')[2][:-7])
                metrics_result['n_sources'].append(int(direct.split('_')[2][:-1]))
                metrics_result['top_k'].append(top_k)
                metrics_result['infection_percentage'].append(int(direct.split('_')[1][:-3]))
                metrics_result['precision_mean'].append(precision_arr.mean())
                metrics_result['recall_mean'].append(recall_arr.mean())
                metrics_result['f_score_mean'].append(f_score_arr.mean())

                metrics_result['precision_var'].append(precision_arr.var())
                metrics_result['recall_var'].append(recall_arr.var())
                metrics_result['f_score_var'].append(f_score_arr.var())

    f = open(f'data/metrics_output/metrics_output_mixed{"_nb" if neighbors_positive else ""}.json', 'w')  # /test
    f.write(json.dumps(metrics_result))


if __name__ == '__main__':
    test_mixed(True)
