import torch

DEVICE = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

EMBEDDING_DIR = 'data/embedding'
INFECTED_DIR = 'data/infected_graph'

TRANSFORMED_DIR = 'data/graph_transformed'
NOT_TRANSFORMED_DIR = 'data/graph_enriched'

GRAPH_ENRICHED = ['data/graph_enriched/er_5inf_1s', 'data/graph_enriched/er_5inf_3s',
                  'data/graph_enriched/er_5inf_5s', 'data/graph_enriched/er_5inf_7s',
                  'data/graph_enriched/er_10inf_3s', 'data/graph_enriched/er_15inf_3s',
                  'data/graph_enriched/er_20inf_3s', 'data/graph_enriched/er_15inf_10s',
                  'data/graph_enriched/er_15inf_15s', 'data/graph_enriched/er_15inf_20s']


TOP_K = {1: [1, 2, 5, 7, 10],
         2: [2, 4, 8, 16, 20],
         3: [3, 6, 12, 24, 30],
         5: [5, 10, 20, 40, 50],
         7: [7, 14, 28, 56, 70],
         10: [10, 20, 40, 80, 100],
         15: [15, 30, 60, 120, 150],
         20: [20, 40, 80, 160, 200]}

MAKE_NEIGHBORS_POSITIVE = False

MODEL_GRAPH_DIR = 'data/model/graph/'
MODEL_CLASS_DIR = 'data/model/classifier'

MODEL_FILE = 'sagemodel.pickle'

GRAPH_SUP_UNTRANS_FILE = 'sage_model_sup_untrans_pooling.pickle'
GRAPH_SUP_UNTRANS_BIN_FILE = 'sage_model_sup_untrans_bin.pickle'
GRAPH_SUP_UNTRANS_BIN_FULL_BATCH_FILE = 'sage_model_sup_untrans_bin_full_batch.pickle'
GRAPH_SUP_UNTRANS_BIN_FULL_2_LAYERS_FILE = 'sage_model_sup_untrans_bin_full_batch_3_layers_improved.pickle'
GRAPH_UNSUP_TRANS_FILE = 'sage_model_unsup_trans.pickle'
GRAPH_UNSUP_UNTRANS_FILE = 'sage_model_unsup_untrans.pickle'
