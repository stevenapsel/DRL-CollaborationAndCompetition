# DRL - Collaboration And Competition

[//]: # (Image References)

![](./tennis.png)

# Project 3: Collaboration and Competition

### Algorithm

describe MADDPG, reference papers, etc

#### Modfications

discuss any modifications and customizations from the papers, etc.
references

### Implementation

The code used for this project is heavily based on the solution in the....

#### model.py:
<b> REVISE THIS </b>
This file contains the network classes for the Actor and the Critic.  The networks use rectified non-linearity (ReLU activation) for all the hidden layers.  The final output layre of the actor used tanh activation to keep the actions bounded.  Both networks had hidden layers of 400 and 300.  Both networks apply Batch Normalization after the first hidden layer.  For the Critic, the actions aren't input to the network until the second hidden layer.  The implemenation is nearly an exact match of the description found in Section 7 (Experiment Details) of the Continuous Control paper.

#### maddpg_agent.py:
describe all the methods

#### Tennis.ipynb
The maddpg() method in the notebook is the main training loop.  It uses Agent.act to ...

### Hyperparameters
The following hyperparameter settings were used:
```
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
LR_ACTOR = 1e-3         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
LEARN_EVERY = 1         # learning timestep interval
LEARN_NUM = 1           # number of learning passes
GAMMA = 0.99            # discount factor
TAU = 7e-2              # for soft update of target parameters
OU_SIGMA = 0.2          # Ornstein-Uhlenbeck noise parameter, volatility
OU_THETA = 0.12         # Ornstein-Uhlenbeck noise parameter, speed of mean reversion
EPS_START = 5.5         # initial value for epsilon in noise decay process in Agent.act()
EPS_EP_END = 250        # episode to end the noise decay process
EPS_FINAL = 0           # final value for epsilon after decay
```
The model architecture for the neural network is described above in the model.py section.

### Plot of Rewards

### Ideas for Future Work

#### Hyperparameter Tuning

#### Network Architecture

what else????
