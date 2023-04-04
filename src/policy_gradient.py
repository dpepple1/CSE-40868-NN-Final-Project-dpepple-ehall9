#BULK OF CODE FROM https://github.com/pytorch/examples/blob/main/reinforcement_learning/reinforce.py


'''
Current issues to figure out:
-   Currently only takes one action per step
-   Neural Network is taking 10240 inputs, 
    one for each step in memory.
'''

import sys


import argparse
import numpy as np
from itertools import count
from collections import deque
import retro

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from neural_network import PGNN

#Training Parameters
EPISODES = 10
MAX_STEPS = 1000
GAMMA = 0.99
RENDER = True
LOG_INTERVAL = 10


env = retro.make(game="GalagaDemonsOfDeath-Nes", obs_type=retro.Observations.RAM)
print("Action Space Shape:", env.action_space)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

policy = PGNN(10240,9)
policy.to(device)
optimizer = optim.Adam(policy.parameters(), lr=0.01)
eps = np.finfo(np.float32).eps.item()

def select_action(state):
    '''
    Neural Network will output percentages for each of the possible actions.
    Decision is made by taking a weighted random sample from those options,
    hence the Categorical() distribution. See Geron 617 for info.
    '''
    state = torch.from_numpy(state).float().unsqueeze(0)
    state = state.to(device)    
    probs = policy(state)
    #print(probs)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action)) 
    return action.item()

def finish_episode():
    R = 0
    policy_loss = []
    returns = deque()
    for r in policy.rewards[::-1]:
        R = r + GAMMA * R
        returns.appendleft(R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    #print(policy_loss)
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]

def main():
    running_reward = 10
    for i_episode in range(EPISODES):
        state= env.reset()        
        ep_reward = 0
        for t in count():  # Don't infinite loop while learning
            action = select_action(state)
            state, reward, done, info = env.step([action])
            if RENDER:
                env.render()
            policy.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        finish_episode()

        print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
            i_episode, ep_reward, running_reward))

        #THIS CODE GOES UNTIL A THRESHOLD REWARD IS HIT 
        #if i_episode % LOG_INTERVAL == 0:
        #    print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
        #          i_episode, ep_reward, running_reward))
        #if running_reward > env.spec.reward_threshold:
        #    print("Solved! Running reward is now {} and "
        #          "the last episode runs to {} time steps!".format(running_reward, t))
        #    break

    
            


if __name__ == '__main__':
    main()