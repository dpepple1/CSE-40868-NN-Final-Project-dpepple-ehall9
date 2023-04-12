# CSE-40868-NN-Final-Project
An experiment in training a neural network using reinforcement learning to play Galaga

## Resources:
[Gym Retro Documentation](https://retro.readthedocs.io/en/latest/index.html)  
[Gym Retro Game Integrator Instructions](https://retro.readthedocs.io/en/latest/integration.html)  
[Video on finding variables with integrator](https://www.youtube.com/watch?v=lPYWaUAq_dY)  

## First Solution Report:

### Report
Features on screen are identified using opencv, as a neural network is not necessary to identify largely static images, and so runs faster while performing reinfocement learning.
A softmax function is used because there are a discrete number of potential movements which the network can choose, so softmax determines which way theship moves, and if it shoots. 
Reinfocement learning requires a reward function, and score is used as it is the common metric of success in competition.  
Since softmax chooses a specific action, the loss function utilizes the reward from the chosen action compared to the probability of choosing another action.
The three linear layers used go from the total of the observation space down to the output the size of the action space, with one possible output per possible action.

### Contributions
Evan: Image recognition using opencv to get enemy and ship locations
Created template images for each enemy and for the player ship
Used templates for matching, and scaled templates to correctly place ship
Package versioning struggles and solutions
