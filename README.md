# GCN Epidemic Source Detector

## Steps:

1) *generate_input.py*: in order to run experiments, it's necessary first to generate infected graphs. Configure graph_config.json with type of network and infection parameters to be generated and run generate_input.py to generate said infected graphs.
2) *transform_graph.py*: uses previously generated graphs infected state to create new node attributes. The attributes are then saved within the graph.  
3) *create_multiple_model_sup_unstrans_mixed.py*, *create_multiple_model_sup_untrans_bin.py*, *create_multiple_model_sup_untransformed.py* -> node_classification
4) 
