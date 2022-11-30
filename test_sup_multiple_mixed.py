import json
import os
import pickle
import numpy as np
import torchmetrics

from utils.constants import TOP_K, MAKE_NEIGHBORS_POSITIVE, MODEL_MIXED, NOT_TRANSFORMED_DIR
from utils.test_model import test_pred


def make_neighborhood_positive(data):
    sources = np.argwhere(data.y > 0)[0]
    neighbors_of_sources = np.array([])
    for source in sources:
        nb_source_idx_0 = data.edge_index[1][np.argwhere(data.edge_index[0] == int(source))[0]]
        neighbors_of_sources = np.append(neighbors_of_sources, nb_source_idx_0)

    y = data.y.clone().detach()
    y[neighbors_of_sources] = 1
    return y


def main():
    metrics_result = {'n_sources': [], 'top_k': [],
                      'precision_mean': [], 'recall_mean': [],
                      'infection_percentage': [], 'f_score_mean': []}
    # Load test graphs
    for sage_model in os.listdir(MODEL_MIXED):
        sage = pickle.load(open(os.path.join(MODEL_MIXED, sage_model), 'rb'))
        infection = int(sage_model.split('_')[1][:len(sage_model.split('_')[1]) - 3])
        sources = sage_model.split('_')[2].split('.')[0][:-1].split('-')

        files = []
        for s in sources:
            files.append(f"{sage_model.split('-')[2][:-1]}{s}s")

        for direct in files:
            for top_k in TOP_K[int(direct.split('_')[2][:-1])]:
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

                    y = data.y if not MAKE_NEIGHBORS_POSITIVE else make_neighborhood_positive(data)

                    # Eval
                    f_score = torchmetrics.functional.f1_score(y_pred, y)

                    roc = torchmetrics.ROC()
                    fpr, tpr, thresholds = roc(y_pred, y)
                    fpr = fpr[1]
                    tpr = tpr[1]

                    fnr = 1 - tpr
                    tnr = 1 - fpr

                    prc = torchmetrics.functional.precision(y_pred, y)
                    rec = torchmetrics.functional.recall(y_pred, y)

                    print(f'N sources pred: {sum(y_pred)}')
                    print(f'FPR: {fpr}, FNR: {fnr}')
                    print(f'TPR: {tpr}, TNR: {tnr}')
                    print(f'Precision: {prc}')
                    print(f'Recall: {rec}')

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

                metrics_result['n_sources'].append(int(direct.split('_')[2][:-1]))
                metrics_result['top_k'].append(top_k)
                metrics_result['infection_percentage'].append(infection)
                metrics_result['precision_mean'].append(precision_arr.mean())
                metrics_result['recall_mean'].append(recall_arr.mean())
                metrics_result['f_score_mean'].append(f_score_arr.mean())

    f = open(f'data/metrics_output_mixed_test{"_nb" if MAKE_NEIGHBORS_POSITIVE else ""}.json', 'w')  # /test
    f.write(json.dumps(metrics_result))


if __name__ == '__main__':
    main()
