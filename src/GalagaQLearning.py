import numpy as np
import retro

import torch
from torch import optim

from MLP import MLP
from Discretizer import GalagaDiscretizer
from imagetest import EnemyFinder

from QAgent import Agent

from utils import plotLearning

if __name__ == '__main__':
    env = retro.make(game="GalagaDemonsOfDeath-Nes", obs_type=retro.Observations.IMAGE)
    env = GalagaDiscretizer(env)

    xres, yres, _ = env.observation_space.shape
    xdiv = xres // 5
    ydiv = yres // 5

    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=env.action_space.n, 
                  eps_end= 0.01, input_dims = [xdiv * ydiv], lr=0.003)
    
    ef = EnemyFinder(xres, yres, xdiv, ydiv)

    scores, eps_history = [], []
    n_games = 500
    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        grid = ef.fill_grid(observation)
        grid = grid.flatten()

        steps = 0

        while not done:
            action = agent.choose_action(grid)
            observation_, reward, done, info = env.step(action)
            #env.render()
            score += reward

            grid_ = ef.fill_grid(observation_)
            grid_ = grid_.flatten()

            agent.store_transition(grid, action, reward, grid_, done)
            agent.learn()
            grid = grid_
            if steps > 10000:
                done = True
            
            steps += 1
            


        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])
        print('episode ', i, 'score %.2f' % score, 
              'average score %.2f' % avg_score,
              'epsilon %.2f' % agent.epsilon)
        
    x = [i+1 for i in range(n_games)]
    filename = 'qgraph.png'
    plotLearning(x, scores, eps_history, filename)

        