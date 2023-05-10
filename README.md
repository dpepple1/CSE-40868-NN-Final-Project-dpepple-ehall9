# CSE-40868-NN-Final-Project
An experiment in training a neural network using reinforcement learning to play Galaga

## Resources:
[Gym Retro Documentation](https://retro.readthedocs.io/en/latest/index.html)  
[Gym Retro Game Integrator Instructions](https://retro.readthedocs.io/en/latest/integration.html)  
[Video on finding variables with integrator](https://www.youtube.com/watch?v=lPYWaUAq_dY)  

## First Solution Report:

### Report
Features on screen are identified using opencv, as a neural network is not necessary to identify largely static images, and so runs faster while performing reinforcement learning. The screen resolution of Galaga is 224px by 240px. To reduce the number of inputs being fed to the neural network, we attempted to reduce the screen to a 22x24 grid. In this grid, locations noted as having an enemy are stored as a 1, locations with the player ship are stored as 2, all other locations are stored as 0. This functionality is handeled by the EnemyFinder class in imagetest.py. 

The neural network being used is a fairly simple MLP with 3 hidden layers. The hidden layers all use ReLU activation functions. For the output layer, a softmax function is used because there are a discrete number of potential movements which the network can choose, so softmax outputs a set of probabilities that sum to 1. This way, the agent can make a weighted random selection for which action to take.

In the original Gym Environement, the action space consists of 9 possible actions, which refer to the different buttons on the NES controller. However, we reduced our action space to 6 possible actions; those being do nothing, fire a missile, move left, move right, fire and move left, and fire and move right. This way the policy could choose just one action instead of having to select muliple actions simultaneously.

We are basing the training of our algorithm on the REINFORCE algorithm, which is a simple reinforcement learning policy gradient technique. Reinfocement learning requires a reward function, and score is used as it is the common metric of success in competition. Since softmax chooses a specific action, the loss function utilizes the reward from the chosen action compared to the probability of choosing another action.

The first solution is able to average a score of roughly 5000 points per run, which is not good for a human player, but is better than the random algorithm could do with any consistency.

For the final solution, there are several problems we would like to fix.
The image recognition has problem with rotation, which can be fixed by adding more rotated templates, but this also slows runtime. The network also does not well recognize the missiles that come out of the enemies, which should also be fixed with the addition of more templates. Template recognition also has some difficulty at the very left edge of the screen, for reasons currently unclear. The network is in the habit of going towards one corner of the screen and sitting there, which makes clearing stages more difficult. This will likely be the hardest to fix, as the ship's aversion to enemies also keeps it alive longer, so it will require taking the reward function significantly to balance risk and reward better. We are also considering the possibility of exploring other methods of reinforcement learning such as Deep Q learning.

### Contributions
Evan: Image recognition using opencv to get enemy and ship locations  
Created template images for each enemy and for the player ship  
Used templates for matching, and scaled templates to correctly place ship  
Package versioning struggles and solutions  

Derek: Configured MLP  
Setup Discretizer to change action space  
Implemented REINFORCE Algorithm from Towards Data Science article  
Implemented other policy gradient code that we eventually gave up on  


## Final Report  
There were sevearl changes made to the code base in the attempt to teach a network to play Galaga. After several changes in image detection, reward function, network architecture and training methods, we found that we were still unable to get a network to play any better than in past attempts. Here are some of the changes we implemented.   

### Image Detection  

### Deep Q Learning
The second major change that we attemped in trying to improve the network was in using Deep Q Learning. In the previous attemps, we were using the REINFORCE policy gradient algorithm to train our network. This involved having the network output a probability for each possible action that it believed would lead to the highest outcome. Alternatively, a Deep Q Network (DQN) attempts to predict the score that would be obtained by the agent for each of the actions in the action space. We call these predictions Q values. When the agent steps once through the environment, the previous guess is compared to the reward that was actually obtained, and the loss is computed. When training, we use a decayed epsilon greedy approach to choosing the action to take based on the network's output. Initially, epsilon starts at 1, meaning that a random action is taken over the networks's predicted highest scoring action 100% of the time. With each step this is decreased until reaching 0.01, where the network now has more control.
 

