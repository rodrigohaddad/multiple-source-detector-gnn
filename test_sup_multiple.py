import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from torch_geometric.utils import accuracy, f1_score, false_positive, true_positive, precision, recall
from sklearn.metrics import f1_score as sk_f1_score

from utils.constants import MODEL_GRAPH_DIR, GRAPH_SUP_UNTRANS_FILE, NOT_TRANSFORMED_DIR, GRAPH_SUP_UNTRANS_BIN_FILE, \
    GRAPH_SUP_UNTRANS_BIN_FULL_2_LAYERS_FILE, GRAPH_ENRICHED
from utils.test_model import test_pred


def main():
    # Load embedding model
    for enriched_path in GRAPH_ENRICHED:
        sage_model_name = f'graph-sage-{enriched_path.split("/")[-1]}.pickle'
        sage = pickle.load(open(f'{MODEL_GRAPH_DIR}{sage_model_name}', 'rb'))
        data_plot = {'FPR': [], 'FNR': [], 'f_score': []}
        index = []
        fpr_arr = np.array([])
        fnr_arr = np.array([])
        f_score_arr = np.array([])
        precision_arr = np.array([])
        recall_arr = np.array([])

        # Load test graphs
        for i, filename in enumerate(os.listdir(f"{NOT_TRANSFORMED_DIR}/test")):
            file = os.path.join(f"{NOT_TRANSFORMED_DIR}/test", filename)
            if not os.path.isfile(file):
                continue

            # Data
            data = pickle.load(open(file, 'rb'))
            # data.y = data.x[:, -1].float()
            data.y = data.x[:, -1].long()
            data.x = data.x[:, :-1].float()

            # Apply embedding model
            y_pred = test_pred(sage, data)

            # Eval
            acc = accuracy(data.y, y_pred)
            f_score = f1_score(y_pred, data.y, 2)
            sk_f = sk_f1_score(data.y, y_pred, average='micro')

            fn, fp = false_positive(y_pred, data.y, 2)
            tn, tp = true_positive(y_pred,  data.y, 2)

            prc = precision(y_pred, data.y, 2)
            rec = recall(y_pred, data.y, 2)

            print(f'N sources pred: {sum(y_pred)}')
            print(f'Acc: {acc}, F_score: {f_score}, {sk_f}')
            print(f'FPR: {fp/(tn+fp)}, FNR: {fn/(tp+fn)}')
            print(f'TPR: {tp / (tp + fn)}, TNR: {tn / (tn + fp)}')
            print(f'Precision: {prc}')
            print(f'Recall: {rec}')

            index.append(f'G_{i}')
            fpr_arr = np.append(fpr_arr, float(fp/(tn+fp)))
            fnr_arr = np.append(fnr_arr, float(fn/(tp+fn)))
            f_score_arr = np.append(f_score_arr, float(f_score[1]))

            precision_arr = np.append(precision_arr, float(prc[0]))
            recall_arr = np.append(recall_arr, float(rec[0]))

            data_plot['FPR'].append(float(fp/(tn+fp)))
            data_plot['FNR'].append(float(fn/(tp+fn)))
            data_plot['F-score'].append(float(f_score[1]))
            data_plot['Precision'].append(float(prc[1]))
            data_plot['Recall'].append(float(rec[1]))

        df = pd.DataFrame({'Precision': data_plot['Precision'],
                           'Recall': data_plot['Recall'],
                           'F-score': data_plot['F-score']},
                          index=index)

        axes = df.plot.bar(rot=0, subplots=True, grid=True,
                           color=['#FEBCC8', '#C8CFE7', '#C7E5C6'],
                           title='Supervised - Test set - 5 sources')
        axes[1].legend(loc=2)
        axes[0].set_title(f'Precision M:{precision_arr.mean():.4f} V:{precision_arr.var():.4f}')
        axes[1].set_title(f'Recall M:{recall_arr.mean():.4f} V:{recall_arr.var():.4f}')
        axes[2].set_title(f'F-score M:{f_score_arr.mean():.4f} V:{f_score_arr.var():.4f}')
        plt.savefig("data/figures/sup_bin_full_batch_test_3l_imp_5", dpi=120)


if __name__ == '__main__':
    main()
