import gymnasium as gym
import retro
import imagetest
from Discretizer import GalagaDiscretizer
from imagetest import EnemyFinder

env = retro.make(game="GalagaDemonsOfDeath-Nes[A] - Copy")
env = GalagaDiscretizer(env)
ef = EnemyFinder()

obs = env.reset()

for i in range(200):
    action = env.action_space.sample()
    #print(action)
    obs, rewards, done, info = env.step(action)
    #imagetest.find_enemy(obs, i)
    ef.find_enemies(obs, i)
    #env.render()

    if done:
        break