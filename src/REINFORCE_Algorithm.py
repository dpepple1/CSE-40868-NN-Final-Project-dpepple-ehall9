import numpy as np
import retro

import torch
from torch import optim

from MLP import MLP
from Discretizer import GalagaDiscretizer
from imagetest import EnemyFinder


#REINFORCE Tutorial From:
#https://towardsdatascience.com/learning-reinforcement-learning-reinforce-with-pytorch-5e8ad7fc7da0

def discount_rewards(rewards, gamma=0.99):
    r = np.array([gamma**i * rewards[i] for i in range(len(rewards))])
    #Reverse the array direction for cumsum and then
    #revert back to original order
    r = r[::-1].cumsum()[::-1]
    return r - r.mean()


def reinforce(env, policy_estimator, enemy_finder, device, num_episodes=2000, batch_size=10, gamma=0.99):
    
    # Set up lists to hold results
    total_rewards = []
    batch_rewards = []
    batch_actions = []
    batch_states = []
    batch_probs = []
    batch_counter = 1

    # Define optimizer
    optimizer = optim.Adam(policy_estimator.parameters(), 
                           lr=0.01)

    policy_estimator.to(device)

    action_space = np.arange(env.action_space.n)
    ep = 0
    while ep < num_episodes:
        s_0 = env.reset()
        step = 0
        states = []
        rewards = []
        actions = []
        probs = []
        done = False
        while not done:
            # Get actions and convert to numpy array
            grid = enemy_finder.fill_grid(s_0)
            grid = torch.tensor(grid.flatten()).float() #not sure whats up with the .float()
            grid = grid.to(device)
            action_probs = policy_estimator(grid).detach().cpu().numpy()
            action = np.random.choice(action_space, p=action_probs)
            s_1, r, done, _ = env.step(action)
            step += 1
            if step > 300:
                done = True
            #env.render()

            states.append(s_0)
            probs.append(action_probs)
            rewards.append(r)
            actions.append(action)

            #If done, batch data
            if done:
                batch_rewards.extend(discount_rewards(rewards, gamma))
                batch_states.extend(states)
                batch_actions.extend(actions)
                batch_probs.extend(probs)
                batch_counter += 1
                total_rewards.append(sum(rewards))
                print('reward', sum(rewards))

                #If batch is complete, updatae network
                if batch_counter == batch_size:
                    optimizer.zero_grad()

                    state_tensor = torch.FloatTensor(np.array(batch_states))
                    reward_tensor = torch.FloatTensor(np.array(batch_rewards))
                    #Actions are used as indicies, must be LongTensor
                    action_tensor = torch.LongTensor(np.array(batch_actions))

                    prob_tensor = torch.FloatTensor(np.array(batch_probs))

                    #Calculate loss
                    #state_tensor = state_tensor.to(device)
                    print(action_tensor)
                    #logprob = torch.log(policy_estimator(state_tensor)).cpu()
                    logprob = torch.log(prob_tensor).T
                    print(logprob)
                    selected_logprobs = reward_tensor * torch.gather(logprob, 1, action_tensor).squeeze()
                    loss = -selected_logprobs.mean()

                    #Calculate gradients
                    loss.backward()
                    #Apply gradients
                    optimizer.step()

                    batch_rewards = []
                    batch_actions = []
                    batch_states = []
                    batch_counter = 1

                avg_rewards = np.mean(total_rewards[-100:])
                #Print running average
                #print("\rEp: {} Average of last 100:" + "{:.2f}".format(ep + 1, avg_rewards), end="")
                ep += 1

    return total_rewards

def main():
    mode = 'train'

    #Setup Gym environment
    env = retro.make(game="GalagaDemonsOfDeath-Nes[A]", obs_type=retro.Observations.IMAGE)
    env = GalagaDiscretizer(env)

    xres, yres, _ = env.observation_space.shape
    xdiv = xres // 10
    ydiv = yres // 10

    #Setup enemy finder 
    ef = EnemyFinder(xres, yres, xdiv, ydiv)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Currently using device: ", device)

    actions = env.action_space.n
    model = MLP(xdiv * ydiv, actions)

    rewards = reinforce(env, model, ef, device, batch_size=5)

if __name__ == '__main__':
    main()