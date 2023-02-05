### SARSA

SARSA (State-Action-Reward-State-Action) is a type of reinforcement learning algorithm used to train agents to interact with an environment. It is an on-policy algorithm, meaning that it learns from the actions the agent takes in the environment. SARSA works by using a reward system to incentivize the agent to take certain actions in order to reach a desired goal.

At each step, the agent takes an action based on the current state of the environment and then receives a reward. The reward is used to update the agent's policy, which helps the agent learn which action to take in a given state. This process is repeated until the agent reaches the desired goal.

It is an on-policy learning algorithm, meaning that it learns from the actions taken by the agent in the environment. It is used to find an optimal policy for an agent by learning from the rewards it receives from its actions.

The equation behind SARSA is as follows:
```
Q(s,a) = Q(s,a) + α[R + γQ(s',a') - Q(s,a)]
```
Where:
```
Q(s,a) = the estimated value of taking action a in state s

α = the learning rate

R = the reward received from taking action a in state s

γ = the discount factor

Q(s',a') = the estimated value of taking action a' in the next state s'
```

A simple use-case example of SARSA could be a robot navigating a maze. The robot would take an action based on its current state (i.e. the position of the walls in the maze) and then receive a reward based on how close it is to the goal. The robot would then use the reward to update its policy and learn which action to take in order to reach the goal. This process would be repeated until the robot reaches the goal.