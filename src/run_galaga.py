import gymnasium as gym
import retro 
import imagetest
from Discretizer import GalagaDiscretizer
from imagetest import EnemyFinder
from MLP import MLP
import numpy as np
import torch

print(__file__)

print(retro.data.__dir__())
print(retro.data.get_romfile_path('GalagaDemonsOfDeath-Nes'))
print(retro.data.add_integrations)

env = retro.make(game="GalagaDemonsOfDeath-Nes")
env = GalagaDiscretizer(env)
ef = EnemyFinder(224, 240, 22, 24)

obs = env.reset()

xres, yres, _ = env.observation_space.shape
xdiv = xres // 10
ydiv = yres // 10


GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
CYAN = '\033[96m'

BASE = '\033[0m'

quit()

model = MLP(xdiv * ydiv, 6)

for i in range(5000):
    action = env.action_space.sample()
    #print(action)
    obs, rewards, done, info = env.step(5)


    grid = ef.fill_grid(obs)    
    grid_str = grid.T.__str__()

    grid_str = grid_str.replace('1', 'A')
    grid_str = grid_str.replace('2', 'B')
    grid_str = grid_str.replace('3', 'C')
    grid_str = grid_str.replace('4', 'D')

    grid_str = grid_str.replace('0', '_')
    grid_str = grid_str.replace('A', GREEN + '1' + BASE)
    grid_str = grid_str.replace('B', RED + '2' + BASE)
    grid_str = grid_str.replace('C', YELLOW + '3' + BASE)
    grid_str = grid_str.replace('D', CYAN + '4' + BASE)

    #print(grid.shape)
    #print(grid_str)

    grid = torch.tensor(grid.flatten()).float()
    action = model(grid)

    env.render()

    if done:
        break