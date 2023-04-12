import retro
from nes_py.wrappers import JoypadSpace
from Discretizer import Discretizer, GalagaDiscretizer

env = retro.make(game='GalagaDemonsOfDeath-Nes[A]')
env = GalagaDiscretizer(env)
print(env.action_space)

print(env.action_space)
obs = env.reset()
for x in range(env.action_space.n):
    print(x)
    obs = env.reset()
    for i in range(1000):
        action = env.action_space.sample()
        #print(action)
        obs, rewards, done, info = env.step(x)
        env.render()

        if done:
            break