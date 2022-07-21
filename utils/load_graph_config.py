from infected_graph_factory.infection_config import GraphConfig
import json


def load_config():
    f = open('data/graph_config.json')
    graph_config = json.load(f)
    f.close()
    return [GraphConfig(**param) for param in graph_config]
