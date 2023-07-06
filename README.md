# GCN Epidemic Source Detector

## Steps:

1) *generate_input.py*: in order to run experiments, it's necessary first to generate infected graphs. Configure graph_config.json with type of network and infection parameters to be generated and run generate_input.py to generate said infected graphs.
2) *transform_graph.py*: uses previously generated graphs infected state to create new node attributes. The attributes are then saved within the graph.  
3) *create_multiple_model_sup_unstrans_mixed.py*, *create_multiple_model_sup_untrans_bin.py*, *create_multiple_model_sup_untransformed.py*: train different models within different datasets. *Mixed* creates a GCN model trained within the same type of network with node features enriched. *Bin* creates a GCN model trained within a specific dataset with the same type of network with node features enriched and infection configuration. *Untransformed* creates a GCN model trained within a specific dataset with the same type of network and infection configuration.
4) *test_sup_multiple.py* and *test_sup_multiple_mixed.py*: the first script test the model within the a specific dataset with the same type of network with node features enriched and infection configuration. The second, tests the model in different types of graphs.
