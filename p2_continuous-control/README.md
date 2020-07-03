

# Continuous Control

### Introduction

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"

For this project, I trained an agent to control an arm to move to a target location in the Unity [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

The task is episodic, and in order to solve the environment,  your agent must get an average score of +30.0 over 100 consecutive episodes.

### Getting Started
The environment is based on [Unity ML-agents](https://github.com/Unity-Technologies/ml-agents).

#### 1: Clone the DRLND Repository
1.  Clone [the DRLND GitHub repository](https://github.com/udacity/deep-reinforcement-learning#dependencies) from Udacity.
2. Follow the instructions in the README to configure the necessary dependencies--Python, PyTorch, Numpy, ML-Agents, etc.

#### 2: Download the Unity Environment
1. Download the environment from one of the links below. You need only select the environment that matches your operating system:
      - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
       - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)
    
	(_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

	(_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the file in the DRLND GitHub repository, in the `p2_continuous-control/` folder, and unzip (or decompress) the file. 

### Explanation
My solution for this environment uses the actor-critic DDPG algorithm with fixed targets (for both actor and critic), soft updates, experienced replay, and added Ornstein–Uhlenbeck noise. The agent is created with four internal networks: a Q network, a deterministic policy network, a target Q network, and a target policy network.

The Q network and the target Q network have identical architectures: 3 fully-connected layers joined by ReLU activation functions and batch normalization layers. The final output is entered into a tanh activation function.

The policy network and target policy network also have identical architectures: 3 fully-connected layers joined by ReLU activation functions with a batch normalization layer before the 2nd fully-connected layer.

The agent uses a discount rate of 0.99.

The agent is trained in a training loop for either 500 episodes (with a max of 10000 timesteps each) or when it reaches an average reward over 100 episodes of 30.0 or greater.