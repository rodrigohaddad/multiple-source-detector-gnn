import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from torch_geometric.utils import accuracy, f1_score, false_positive, false_negative, true_positive, true_negative
from sklearn.metrics import f1_score as sk_f1_score

from utils.constants import MODEL_GRAPH_DIR, GRAPH_SUP_UNTRANS_FILE, NOT_TRANSFORMED_DIR
from utils.test_model import test_pred


def main():
    # Load embedding model
    sage = pickle.load(open(f'{MODEL_GRAPH_DIR}{GRAPH_SUP_UNTRANS_FILE}', 'rb'))
    data_plot = {'FPR': [], 'FNR': [], 'f_score': []}
    index = []

    # Load test graphs
    for i, filename in enumerate(os.listdir(f"{NOT_TRANSFORMED_DIR}/test")):
        file = os.path.join(f"{NOT_TRANSFORMED_DIR}/test", filename)
        if not os.path.isfile(file):
            continue

        # Data
        data = pickle.load(open(file, 'rb'))
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

        print(f'Acc: {acc}, F_score: {f_score}, {sk_f}')
        print(f'FPR: {fp/(tn+fp)}, FNR: {fn/(tp+fn)}')
        print(f'TPR: {tp / (tp + fn)}, TNR: {tn / (tn + fp)}')
        index.append(f'G_{i}')
        data_plot['FPR'].append(float(fp/(tn+fp)))
        data_plot['FNR'].append(float(fn/(tp+fn)))
        data_plot['f_score'].append(float(f_score[1]))

    df = pd.DataFrame({'FPR': data_plot['FPR'],
                       'FNR': data_plot['FNR'],
                       'f_score': data_plot['f_score']},
                      index=index)

    axes = df.plot.bar(rot=0, subplots=True, grid=True,
                       color=['#FEBCC8', '#C8CFE7', '#C7E5C6'],
                       title='Supervised - Test set')
    axes[1].legend(loc=2)
    plt.savefig("data/figures/sup_2", dpi=120)


if __name__ == '__main__':
    main()
