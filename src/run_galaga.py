import os
import gymnasium as gym
import retro
import imagetest
from Discretizer import GalagaDiscretizer
from imagetest import EnemyFinder
from MLP import MLP
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
print(SCRIPT_DIR)
retro.data.Integrations.add_custom_path(os.path.join(SCRIPT_DIR, "custom_integrations"))
env = retro.make(game="Galaga (U)", inttype=retro.data.Integrations.ALL)
env = GalagaDiscretizer(env)
ef = EnemyFinder(224, 240, 22, 24)

obs = env.reset()

#xres, yres, _ = env.observation_space.shape
#xdiv = xres // 10
#ydiv = yres // 10



while True:
    action = env.action_space.sample()
    #print(action)
    obs, rewards, done, info = env.step(0)
    #print(info)
    if rewards != 0:
        print(rewards)

    #grid = ef.fill_grid(obs)
    #grid = torch.tensor(grid.flatten()).float()
    #print(grid)
    #print(action)

    env.render()

    if done:
        break