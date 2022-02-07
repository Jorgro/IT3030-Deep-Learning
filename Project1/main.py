import json
from network import NeuralNetwork

# TODO:
# - Fix listed configs

if __name__ == "__main__":
    print("Loading config.")
    with open("Project1/configs/config_3.json") as f:
        config = json.load(f)
    nn = NeuralNetwork(config)
    print("Loading dataset")
    nn.load_data()
    print("Training network")
    nn.train()
    print(nn)

# lr=1 for Hidden=4
