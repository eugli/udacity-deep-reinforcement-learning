# Report

## Algorithm

The algorithm used for this environment is [MADDPG](https://arxiv.org/pdf/1706.02275.pdf). MADDPG involves agents that each have an actor network and a critic network. Each agent is independent of the other, but they must learn how the actions of the other agents can potentially affect the environment in order to maximize expected reward. The actor network takes in the current state and outputs an action set for that state. The critic takes in the current state and the action set from the actor and returns the estimated Q-value of the state-action pair. This estimate is then used by the actor network to evaluate its choice of action.

This implementation also employs fixed targets (with soft updates) and experienced replay that helps the agent generalize and train with reduced noise. The fixed targets are stored in two more networks in the agent: a target actor network and target critic network that have identical architectures to the original actor network and target network.

To encourage exploration, the [Ornstein -Uhlenbeck process](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process) is used to add a little bit of noise to the actions.

|Name|Value|
|---|---:|
|Episodes|2000|
|Actor Learning Rate|0.001|
|Critic Learning Rate|0.001|
|Weight Decay|0|
|Gamma|0.99|
|Tau|0.001|
|Buffer Size|100000|
|Batch Size|128|
|Goal|30.0|

#### Model Architecture

The actor (and the actor target) network uses 3 fully-connected layers and 2 batch normalization layers:

- state_size -> 128 -> batch_norm -> ReLU
- 128 -> 128 -> batch_norm -> ReLU
- 128 -> action_size -> tanh

The critic (and the critic target) network uses 3 fully-connected layers and 1 batch normalization layer:

-   state_size -> 128 -> batch_norm -> ReLU
-   128 + action_size -> 128 -> ReLU
-   128 -> 1

The batch normalization layers helps restandardize the input into each fully-connected layer, reducing the internal covariate shift between layers and thus improving training speed.

A convolutional net is not used as the agent does not learn directly from the pixels of the environment but a prepared vector of relevant information.

## Performance

The agent solved the environment (by reaching an average reward of 0.5 over 100 episodes) in  **1510**  episodes, before the 2000 episode limit.

### Reward vs. Episode

![reward](https://user-images.githubusercontent.com/39870221/86493216-a01b7f00-bd3e-11ea-8ebe-ef9133c5c341.png)

## Improvements
- Using prioritized experience replay and/or shared experience replay
- Use other algorithms that are designed for multiple copies of the same agent (like PPO, A3C, and D4PG)
- Adding more fully-connected layers / more hidden units to the layers
- Testing different activation functions (like Leaky ReLU)
- Training past the target average reward of 0.5 and seeing what the limit is
- Optimize hyperparameters (particularly the actor and critic learning rates) using grid search
- Limit how many times the model is updated each timestep to avoid too much noise