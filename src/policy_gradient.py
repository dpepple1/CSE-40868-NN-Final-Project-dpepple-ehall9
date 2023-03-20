import gymnasium as gym
import retro

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from neural_network import PGNN

#Training Parameters
EPISODES = 10
MAX_STEPS = 1000


env = retro.make(game="GalagaDemonsOfDeath-Nes")

policy = PGNN()
optimizer = optim.Adam(policy.parameters(), lr=0.001)


def select_action(state):
    '''
    Neural Network will output percentages for each of the possible actions.
    Decision is made by taking a weighted random sample from those options,
    hence the Categorical() distribution. See Geron 617 for info.
    '''
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action)) 
    return action.item()
  
def main():
    for episode in range(EPISODES):
        obs = env.reset()
        for step in range(MAX_STEPS):
            


if __name__ == '__main__':
    main()