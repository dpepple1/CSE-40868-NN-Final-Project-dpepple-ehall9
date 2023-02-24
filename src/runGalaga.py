import gymnasium as gym
import retro
env = retro.make(game="GalagaDemonsOfDeath-Nes")

observation = env.reset()

for _ in range(1000000):
    action = env.action_space.sample() 
    observation, reward, terminated, truncated = env.step(action)
    env.render()
    if terminated or truncated:
        observation = env.reset()

env.close()