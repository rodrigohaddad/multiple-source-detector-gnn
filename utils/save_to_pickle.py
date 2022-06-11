import pickle


def save_to_pickle(obj, dire, name):
    pickle.dump(obj, open(f"data/{dire}/{name}.pickle", "wb"))
