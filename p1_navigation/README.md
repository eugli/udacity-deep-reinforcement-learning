[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Navigation

### Introduction

For this project, I trained an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Getting Started

The environment is based on [Unity ML-agents](https://github.com/Unity-Technologies/ml-agents)

#### 1: Clone the DRLND Repository
1.  Clone [the DRLND GitHub repository](https://github.com/udacity/deep-reinforcement-learning#dependencies) from Udacity.
2. Follow the instructions in the README to configure the necessary dependencies--Python, PyTorch, Numpy, ML-Agents, etc.

#### 2: Download the Unity Environment
1. Download the environment from one of the links below. You need only select the environment that matches your operating system:
    
    -   Linux:  [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    -   Mac OSX:  [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    -   Windows (32-bit):  [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    -   Windows (64-bit):  [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    Then, place the file in the p1_navigation/ folder in the DRLND GitHub repository, and unzip (or decompress) the file.
    
    (_For Windows users_) Check out  [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64)  if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.
    
    (_For AWS_) If you'd like to train the agent on AWS (and have not  [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use  [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip)  to obtain the environment.
    
### Explanation
My solution for this environment is a standard DQN with fixed q-targets and experienced replay. The agent is created with an internal q-network of 3 fully-connected layers, joined by ReLU activation functions. The agent also satisfies GLIE conditions and uses a discount rate of 0.995.

The agent is trained in a training loop for either 2000 episodes (with a max of 500 timesteps) or when it reaches an average reward of 13.0 or greater.
