import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(MLP, self).__init__()

        #Dense layers
        self.layer1 = nn.Linear(in_features=n_observations, out_features=128)
        self.layer2 = nn.Linear(128, 32)
        self.layer3 = nn.Linear(in_features=32, out_features=n_actions)
                
        #Activation functions
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax() #Softmax in place of sigmoid
        

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        x = self.softmax(x)
        return x 