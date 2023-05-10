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
There were several changes made to the code base in the attempt to teach a network to play Galaga. After several changes in image detection, reward function, network architecture and training methods, we found that we were still unable to get a network to play any better than in past attempts. Here are some of the changes we implemented. Accounting for enemy rotation was handled by switching out detection mode entirely. Enemies are now detected using vectors, which means that any colors that stand out, such as the stars or the enemies are picked up as vectors. 

### Image Detection  
The principal problem with image detection was its challenge in spotting rotations, and the fact the it does not detect the missiles. Detecting both enemy and ship missiles was easy, given that it just had to use more template images. Enemy missiles frequently move at angles, but they do not rotate, meaning that tracking could be done for the projectiles the same way as for the ship. Enemies are now detected using vectors, which means that any colors that stand out, such as the stars or the enemies are picked up as vectors. By selecting vectors over a certain size, stars are eliminated, so all that remains in the window, which is resized to avoid the side, is the enemies, ship, and projectiles. Detection of the ship and projectiles is then overwritten by the value of either the ship or friendly or enemy projectiles. With this method, the grid is able to detect enemies both more accurately and across multiple squares.

### Deep Q Learning
The second major change that we attemped in trying to improve the network was in using Deep Q Learning. In the previous attemps, we were using the REINFORCE policy gradient algorithm to train our network. This involved having the network output a probability for each possible action that it believed would lead to the highest outcome. Alternatively, a Deep Q Network (DQN) attempts to predict the score that would be obtained by the agent for each of the actions in the action space. We call these predictions Q values. When the agent steps once through the environment, the previous guess is compared to the reward that was actually obtained, and the loss is computed. When training, we use a decayed epsilon greedy approach to choosing the action to take based on the network's output. Initially, epsilon starts at 1, meaning that a random action is taken over the networks's predicted highest scoring action 100% of the time. With each step this is decreased until reaching 0.01, where the network now has more control.
 
### Reward Function  
One of the other area we explored was changing the reward function. As a review, the previous reward function was tied directly to the score. Generally, killing an enemy on the screen nets the player either 100, 160, or 200 points in game. This was the value we were using directly for the reward function. First, we tried to penalize death. We did this by adding a -1000 score penalty for dying. This did not seem to improve the gameplay much at all. Then, we noticed that the network seemed to like moving the ship all the way towards the side of the screen. When the network did this, it tended to last a long time before dying (since enemies rarely fly or shoot into the corners) but it also was not able to score many points from this position. So, we attempted to add a penalty for when it stayed in the same location for too long. Strangely, this seemed to make the network stay in the corner even more than before, so we got rid of this change. 

### Hyperparameters and I/O
We also tried changing some hyperparameters, notably learning rate and the epsilon decay value. From what we could find, we believed that both of these values were too high in our initial testing. However, after significantly decreasing both of these values we did not find much, if any, notable difference in performance. We also tried changinig the size of the grid that we were using to represent the system. Increasing the grid size to allow for more definition did not seem to improve much. Finally, we briefly tried reducing the action space from 6 actions to 2 (left and shoot or right and shoot.) This also did not seem to accomplish much.


### Labor Divison

Derek: Implemented DQN and Deep Q Training Agent
Added pentaly for same location
Tested some permutations of hyperparameters
Tested some permutations of grid size and actions space

Evan: Implemented vector based vision approach
Added pentaly for death
Resolved issues regarding custom game integration
Tested some permutations of hyperparameters


### Labor Divison

Derek: Implemented DQN and Deep Q Training Agent
Added penalty for same location
Tested some permutations of hyperparameters
Tested some permutations of grid size and actions space

Evan: Implemented vector based vision approach
Added penalty for death
Resolved issues regarding custom game integration
Tested some permutations of hyperparameters

### Accuracy
We averaged between 5,500-6,000 points per game, which is not great for a human player, but is better than both a random algorithm and slightly better than our previous network.

### Conclusion
