
# Report

## Algorithm

The algorithm used for this environment is [Deep Q-Learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf). This implementation also employs fixed q-targets (through soft updates) and experienced replay that helps the agent generalize and train with reduced noise.

#### Hyperparameters

|Name|Value|
|---|---:|
|Episodes|2000|
|Epsilon Start|1.0|
|Epsilon Decay|0.95|
|Epsilon Min|0.01|
|Learning Rate|0.0005|
|Gamma|0.995|
|Tau|0.001|
|Buffer Size|100000|
|Batch Size|64|
|Goal|13.0|

#### Model Architecture

The model uses 3 fully-connected layers:
- state_size -> 256 -> ReLU
- 256 -> 256 -> ReLU
- 256 -> action_size

A convolutional net is not used as the agent does not learn directly from the pixels of the environment, but a prepared vector of relevant information.

## Performance
The agent solved the environment (by reaching an average reward of 13 over 100 episodes) in **675** episodes, before the 2000 episode limit.

### Reward vs. Episode

![reward](https://user-images.githubusercontent.com/39870221/85939819-6aae0600-b8e6-11ea-9748-cbffbd4b62aa.png)


## Improvements

- Using a double DQN
- Using a dueling DQN
- Using prioritized experience replay
- Using a distributional DQN
- Using a noisy DQN
- Using a (rainbow) combination of the improvements listed above
- Learning form multi-step bootstrapping targets, like in A3C and A2C
- Adding more fully-connected layers / more hidden units to the layers
- Training past the target average reward of 13 and seeing what the limit is
