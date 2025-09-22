# my_module.py
class MyModel:
    def __init__(self, hidden_dim: int = 32, dropout: float = 0.1):
        self.hidden_dim = hidden_dim
        self.dropout = dropout

    def __repr__(self):
        return f"MyModel(hidden_dim={self.hidden_dim}, dropout={self.dropout})"


# run.py
from jsonargparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_class_arguments(MyModel, "model")
    parser.add_argument("--config", action="config")
    cfg = parser.parse_args()
    model = parser.instantiate_classes(cfg).model
    
    print("Instantiated:", model)

if __name__ == "__main__":
    main()