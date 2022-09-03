import torch

DEVICE = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

EMBEDDING_DIR = 'data/embedding'
INFECTED_DIR = 'data/infected_graph'

TRANSFORMED_DIR = 'data/graph_transformed'
NOT_TRANSFORMED_DIR = 'data/graph_not_transformed'

MODEL_GRAPH_DIR = 'data/model/graph'
MODEL_CLASS_DIR = 'data/model/classifier'

MODEL_FILE = 'sagemodel.pickle'

GRAPH_SUP_UNTRANS_FILE = 'sage_model_sup_untrans.pickle'
GRAPH_UNSUP_TRANS_FILE = 'sage_model_unsup_trans.pickle'
GRAPH_UNSUP_UNTRANS_FILE = 'sage_model_unsup_untrans.pickle'

GLOBAL_MODEL_FILE = 'sagemodel_weightless.pickle'
