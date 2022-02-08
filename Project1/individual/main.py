import json
from nn import NeuralNetwork

if __name__ == "__main__":
    print("Loading config.")
    with open("jostein/config_3_2.json") as f:
        config = json.load(f)
    nn = NeuralNetwork(config)
    print("Loading dataset")
    nn.load_data()
    print("Training network")
    nn.train()
    print(nn)
