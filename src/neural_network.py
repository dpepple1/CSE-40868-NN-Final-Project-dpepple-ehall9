import torch
import torch.nn as nn

class PGNN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(PGNN, self).__init__()

        #Dense layers
        self.layer1 = nn.Linear(in_features=n_observations, out_features=20)
        self.layer2 = nn.Linear(20, 20)
        self.layer3 = nn.Linear(in_features=20, out_features=n_actions)
                
        #Activation functions
        self.elu = nn.ELU()
        self.softmax = nn.Softmax(dim=1) #Softmax in place of sigmoid
        self.sigmoid = nn.Sigmoid()

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.layer1(x)
        x = self.elu(x)
        x = self.layer2(x)
        x = self.elu(x)
        x = self.layer3(x)
        x = self.sigmoid(x)
        return x 