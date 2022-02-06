import json
from network import NeuralNetwork

# TODO:
# - PREP FOR DEMO
# - Read through delivery and make sure everything is there!
# - Double check Cross entropy loss function
# - Verbose flag
# - Validate and test to plot
if __name__ == "__main__":
    print("Loading config.")
    with open("Project1/config.json") as f:
        config = json.load(f)
    nn = NeuralNetwork(config)
    print("Loading dataset")
    nn.load_data()
    print("Training network")
    nn.train()
