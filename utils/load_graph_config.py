from infected_graph_factory.infection_config import InfectionConfig
import json


def load_config():
    f = open('../data/graph_config.json')
    inf_config = json.load(f)
    f.close()
    return [InfectionConfig(**param) for param in inf_config]
