import json
from network import NeuralNetwork

# TODO:
# - Add dynamic loss function
# - Data generation
# - Cross-entropy loss function

if __name__ == "__main__":
    print("Loading config.")
    with open("Project1/config.json") as f:
        config = json.load(f)
    print(config)
    print("Running main application")
    nn = NeuralNetwork(config)
    print(nn)
    nn.load_data()
    nn.train()
    # print(nn)
