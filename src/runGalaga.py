import gymnasium as gym
import retro
env = retro.make(game="GalagaDemonsOfDeath-Nes")

observation = env.reset()

while True:
    action = env.action_space.sample() 
    observation, reward, terminated, truncated = env.step(action)
    env.render()
    if terminated or truncated:
        observation = env.reset()

env.close()