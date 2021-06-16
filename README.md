# Udacity-DRL_Collaboration_and_Competition
This repo is an implementation to the final project, called Collaboration and Competition, in the [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893)
<p align="center"><img src=tennis.png></p>

## Project Description 
In this environment, two agents control rackets to bounce a ball over a net. The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Thus, the goal of each agent is to keep the ball in play.

### State Space
The state (observation) space consists of 8 variables corresponding to the position and velocity of the ball and racket, and 3 frames are stacked together for each state, and since two agents exist so the state space is of shape (2,24). Each agent receives its own, local observation.

### Action space
Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.(per agent).

### Rewards
If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. 

### Project Environment
#### Step 1: Clone the [DRLND repo](https://github.com/udacity/deep-reinforcement-learning). Follow the instructions in the readme file to configure the python (3.6 was used in this project) environment.
#### Step 2: Download the Unity Environment according to your operating system:
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
Then, place the file in the p3_collab-compet/ folder in the DRLND GitHub repository, and unzip (or decompress) the file.

#### Step 3: Run
In the terminal run the following command (make sure you are in the correct directory):
```shell
$ jupyter notebook
```
This will open jupyter notebook in the browser, in which you can add the files from this repository to be able to run the code implemented here. 
- Tennis_F.ipynb contains the code that you should run to train the agents.
- model_F.py contains the Actor and Critic Networks.
- maddpg_F.py contains the implementation of the agent step, act and learn, as well as the replay buffer and ounoise.
- checkpoint files contain the trained models that can be loaded and used directly.
