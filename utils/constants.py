import torch

DEVICE = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

EMBEDDING_DIR = 'data/embedding'
INFECTED_DIR = 'data/infected_graph'

TRANSFORMED_DIR = 'data/graph_transformed'
NOT_TRANSFORMED_DIR = 'data/graph_not_transformed_3s_10inf'

MODEL_GRAPH_DIR = 'data/model/graph/'
MODEL_CLASS_DIR = 'data/model/classifier'

MODEL_FILE = 'sagemodel.pickle'

GRAPH_SUP_UNTRANS_FILE = 'sage_model_sup_untrans_pooling.pickle'
GRAPH_SUP_UNTRANS_BIN_FILE = 'sage_model_sup_untrans_bin.pickle'
GRAPH_SUP_UNTRANS_BIN_FULL_BATCH_FILE = 'sage_model_sup_untrans_bin_full_batch.pickle'
GRAPH_SUP_UNTRANS_BIN_FULL_2_LAYERS_FILE = 'sage_model_sup_untrans_bin_full_batch_3_layers_improved.pickle'
GRAPH_UNSUP_TRANS_FILE = 'sage_model_unsup_trans.pickle'
GRAPH_UNSUP_UNTRANS_FILE = 'sage_model_unsup_untrans.pickle'
