
# Report

## Algorithm

The algorithm used for this environment is  [DDPG](https://arxiv.org/pdf/1509.02971.pdf). This implementation also employs fixed targets (with soft updates) and experienced replay that helps the agent generalize and train with reduced noise. To encourage exploration, the [Ornstein -Uhlenbeck process](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process) is used to add a little bit of noise to the actions. 

|Name|Value|
|---|---:|
|Episodes|1000|
|Actor Learning Rate|0.0001|
|Critic Learning Rate|0.0001|
|Weight Decay|0|
|Gamma|0.99|
|Tau|0.001|
|Buffer Size|100000|
|Batch Size|128|
|Goal|30.0|

#### Model Architecture

The actor (and the actor target) network uses 3 fully-connected layers:

- state_size -> 128 -> ReLU
- 128 -> 128 -> ReLU
- 128 -> action_size -> tanh

The critic (and the critic target) network uses 3 fully-connected layers:

-   state_size -> 128 -> batch_norm -> ReLU
-   128 + action_size -> 128 -> ReLU
-   128 -> 1

A convolutional net is not used as the agent does not learn directly from the pixels of the environment but a prepared vector of relevant information.

## Performance

The agent solved the environment (by reaching an average reward of 30.0 over 100 episodes) in  **25**  episodes, before the 1000 episode limit.

### Reward vs. Episode

![reward](https://user-images.githubusercontent.com/39870221/86488820-49a74400-bd30-11ea-8c87-b1abccebcef3.png)

## Improvements
- Using prioritized experience replay
- Adding batch normalization layers
- Testing different activation functions (like Leaky ReLU)
- Adding more fully-connected layers / more hidden units to the layers
- Training past the target average reward of 30.0 and seeing what the limit is
- Optimize hyperparameters (particularly the actor and critic learning rates) using grid search
- Limit how many times the model is updated each timestep to avoid too much noise