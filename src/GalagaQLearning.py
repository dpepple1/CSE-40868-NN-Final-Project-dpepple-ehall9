import os
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
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    print(SCRIPT_DIR)
    retro.data.Integrations.add_custom_path(os.path.join(SCRIPT_DIR, "custom_integrations"))
    env = retro.make(game="Galaga (U)", inttype=retro.data.Integrations.ALL)
    env = GalagaDiscretizer(env)
    max_score = 0
    best_model_path = 'src\\Models\\best_model.pth'
    xres, yres, _ = env.observation_space.shape
    xdiv = xres // 5
    ydiv = yres // 5

    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=env.action_space.n, 
                  eps_end= 0.01, input_dims = [xdiv * ydiv], lr=.00001)
    
    ef = EnemyFinder(xres, yres, xdiv, ydiv)

    scores, eps_history = [], []
    n_games = 500
    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        grid = ef.fill_grid(observation)
        #print(grid.shape)
        grid = grid.flatten()

        steps = 0

        same_place = 0
        prev_place = None

        while not done:
            action = agent.choose_action(grid)


            '''
            try:
                ship_pos = np.where(grid==2)[0][0]
                if ship_pos == prev_place:
                    same_place += 1
                else:
                    same_place = 0
            except:
                #print('ERROR:::', end="")
                pass
            '''
                
            observation_, reward, done, info = env.step(action)


            if same_place > 15:
                reward -= 10 * (same_place - 15)
                #print(f'penalty! : {same_place} {reward}')


            env.render()
            score += reward

            grid_ = ef.fill_grid(observation_)
            grid_ = grid_.flatten()

            agent.store_transition(grid, action, reward, grid_, done)
            agent.learn()
            grid = grid_
            
            steps += 1

            #prev_place = ship_pos
            


        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])
        print('episode ', i, 'score %.2f' % score, 
              'average score %.2f' % avg_score,
              'epsilon %.2f' % agent.epsilon)
        if score > max_score:
            max_score = score
            print('Saving model...')
            torch.save(agent.Q_eval, best_model_path)

    x = [i+1 for i in range(n_games)]
    filename = 'qgraph.png'
    plotLearning(x, scores, eps_history, filename)

        