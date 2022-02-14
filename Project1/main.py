import json
from network import NeuralNetwork

if __name__ == "__main__":
    print("Loading config.")
    with open("Project1/configs/config_1.json") as f:
        config = json.load(f)
    nn = NeuralNetwork(config)
    print("Loading dataset")
    nn.load_data()
    print("Training network")
    nn.train()
    print(nn)
