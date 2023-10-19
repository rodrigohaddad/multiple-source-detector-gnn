# GCN Epidemic Source Detector
An algorithm that uses enriched node information from epidemic infected graphs to train a GCN that is later used to detect epidemic sources.

## Steps:
For quick and simple execution, run the main function. The algorithm is composed by:
1) *generate_input.py*: in order to run experiments, it's necessary first to generate infected graphs. Configure graph_config.json with type of network and infection parameters to be generated and run generate_input.py to generate said infected graphs.
2) *transform_graph.py*: uses previously generated graphs infected state to create new node attributes. The attributes are then saved within the graph.  
3) *create_multiple_model_sup_unstrans_mixed.py*, *create_multiple_model_sup_untrans_bin.py*, *create_multiple_model_sup_untransformed.py*: train different models within different datasets. *Mixed* creates a GCN model trained within the same type of network with node features enriched. *Bin* creates a GCN model trained within a specific dataset with the same type of network with node features enriched and infection configuration. *Untransformed* creates a GCN model trained within a specific dataset with the same type of network without enriched features and infection configuration.
4) *test_sup_multiple.py* and *test_sup_multiple_mixed.py*: the first script test the model within the a specific dataset with the same type of network with node features enriched and infection configuration. The second, tests the model in different types of graphs.
5) The remaining scripts are used for graph analysis.
   

For further details on implementation, please refer to the article.

## Cite

Please cite [the paper](https://link.springer.com/chapter/10.1007/978-3-031-45392-2_22) (and the respective papers of the methods used) if you use this code in your own work:

```
@InProceedings{10.1007/978-3-031-45392-2_22,
author="Haddad, Rodrigo Gon{\c{c}}alves
and Figueiredo, Daniel Ratton",
editor="Naldi, Murilo C.
and Bianchi, Reinaldo A. C.",
title="Detecting Multiple Epidemic Sources in Network Epidemics Using Graph Neural Networks",
booktitle="Intelligent Systems",
year="2023",
publisher="Springer Nature Switzerland",
address="Cham",
pages="331--345",
abstract="Epidemics start within a network because of the existence of epidemic sources that spread information over time to other nodes. Data about the exact contagion pattern among nodes is often not available, besides a simple snapshot characterizing nodes as infected, or not. Thus, a fundamental problem in network epidemic is identifying the set of source nodes after the epidemic has reached a significant fraction of the network. This work tackles the multiple source detection problem by using graph neural network model to classify nodes as being the source of the epidemic. The input to the model (node attributes) are novel epidemic information in the k-hop neighborhoods of the nodes. The proposed framework is trained and evaluated under different network models and real networks and different scenarios, and results indicate different trade-offs. In a direct comparison with prior works, the proposed framework outperformed them in all scenarios available for comparison.",
isbn="978-3-031-45392-2"
}

```
