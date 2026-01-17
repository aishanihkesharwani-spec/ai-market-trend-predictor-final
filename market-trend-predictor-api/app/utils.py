import pickle

def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)
