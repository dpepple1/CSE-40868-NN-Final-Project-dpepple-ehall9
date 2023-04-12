import gymnasium as gym
import retro
import imagetest
from Discretizer import GalagaDiscretizer
from imagetest import EnemyFinder
from MLP import MLP
import numpy as np
import torch

env = retro.make(game="GalagaDemonsOfDeath-Nes[A] - Copy")
env = GalagaDiscretizer(env)
ef = EnemyFinder(224, 240, 22, 24)

obs = env.reset()

xres, yres, _ = env.observation_space.shape
xdiv = xres // 10
ydiv = yres // 10


model = MLP(xdiv * ydiv, 6)

for i in range(200):
    action = env.action_space.sample()
    #print(action)
    obs, rewards, done, info = env.step(5)
    #imagetest.find_enemy(obs, i)
    grid = ef.fill_grid(obs)
    grid = torch.tensor(grid.flatten()).float()
    #print(grid)
    action = model(grid)
    print(action)

    env.render()

    if done:
        break