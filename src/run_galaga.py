import gymnasium as gym
import retro
import imagetest
env = retro.make(game="GalagaDemonsOfDeath-Nes")

obs = env.reset()
for i in range(200):
    action = env.action_space.sample()
    print(action)
    obs, rewards, done, info = env.step(action)
    imagetest.find_enemy(obs, i)
    env.render()

    if done:
        break