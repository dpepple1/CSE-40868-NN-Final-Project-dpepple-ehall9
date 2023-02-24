import gymnasium as gym
import retro
env = retro.make("GalagaDemonsOfDeath-Nes")
observation = env.reset()

for _ in range(1000):
    action = env.action_space.sample() 
    observation, reward, terminated, truncated = env.step(action)

    if terminated or truncated:
        observation = env.reset()

env.close()