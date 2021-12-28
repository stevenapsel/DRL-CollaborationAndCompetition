# DRL-CollaborationAndCompetition
Udacity project #3
## Project Goal
The goal of this project is to train agents that control tennis rackets.  We want the agents to be able to hit a ball back and forth over a net.  The goal is collaborative in that both agents are trying to keep the "rally" going by keeping the ball in play (rather than trying to beat each other).

![](./tennis.png)

## Environment Details

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The environment actually reports a stack of 3 consecutive observations (frames), so the observed state space for an agent is 24.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
This yields a single score for each episode.
The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.
