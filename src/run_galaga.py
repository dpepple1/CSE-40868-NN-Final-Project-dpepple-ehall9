import gymnasium as gym
import retro
env = retro.make(game="GalagaDemonsOfDeath-Nes")

obs = env.reset()
while True:
    action = env.action_space.sample()
    print(action)
    obs, rewards, done, info = env.step(action)
    env.render()

    if done:
        break