import json
from network import NeuralNetwork

# TODO:
# - PREP FOR DEMO
# - Read through delivery and make sure everything is there!
# - Double check Cross entropy loss function
# - Verbose flag

if __name__ == "__main__":
    print("Loading config.")
    with open("Project1/configs/sigmoid_mse_reg.json") as f:
        config = json.load(f)
    print(config)
    print("Running main application")
    nn = NeuralNetwork(config)
    print(nn)
    nn.load_data()
    nn.train()
    # print(nn)
