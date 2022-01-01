# DRL - Collaboration And Competition

[//]: # (Image References)

![](./tennis.png)

# Project 3: Collaboration and Competition

### Algorithm

For this project I implemented a variant of the Multi-Agent Deep Deterministic Policy Gradient (MADDPG) algorithm as described in the paper: [Multi-Agent Actor-Critic for Mixed
Cooperative-Competitive Environments](https://proceedings.neurips.cc/paper/2017/file/68a9750337a418a86fe06c1991a1d64c-Paper.pdf)

MADDPG is an extension of Deep Deterministic Policy Gradient (DDPG) algorithm described in the paper: [Continuous Control With Deep Reinforcement Learning](https://arxiv.org/abs/1509.02971)

We used DDPG in Project #2 when we were training a single agent in a continous action space.  In project #3, we are training two agents that need to act in collaborative manner.  MADDPG is well-suited for this latter use case.  It extends DDPG by modifying the data available to the critic during training such that the critic can know the actions performed by <b>all</b> agents at each step:

> We adopt the framework of centralized training with decentralized execution, allowing the policies
> to use extra information to ease training, so long as this information is not used at test time. It is
> unnatural to do this with Q-learning without making additional assumptions about the structure of the
> environment, as the Q function generally cannot contain different information at training and test
> time. Thus, we propose a simple extension of actor-critic policy gradient methods where the critic is
> augmented with extra information about the policies of other agents, while the actor only has access
> to local information. After training is completed, only the local actors are used at execution phase,
> acting in a decentralized manner and equally applicable in cooperative and competitive settings. This
> is a natural setting for multi-agent language learning, as full centralization would not require the
> development of discrete communication protocols.

The extra information is only used by the critic during training.  The inference performed by the actor (to select actions) uses the same local observations as DDPG.  

The multi-agent decentralized actor, centralized critic approach can be seen in this figure from the paper:

![](./multi-agent-actor-critic.png)

See how more than one green arrow is input to critic?  That's the key extension from DDPG to MADDPG.

#### Modfications

For project #2, I came upon a modification to the original DDPG at [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/algorithms/ddpg.html)

This implementation incorporated the following key features that yielded immediate improvements:
* A learning step to better control the update of the networks relative to the amount of experiences collected
* Gradient clipping in the critic network
* A decay term to gradually reduce the introduction of noise as training progresses

For this project, we modified the decay, to put a limit on the number of episodes for which exploration will be allowed.

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
