# SARSA Learning Algorithm
## AIM
The bandit slippery walk problem is a reinforcement learning problem in which an agent must learn to navigate a 7-state environment in order to reach a goal state. 
## PROBLEM STATEMENT
Train agent with SARSA in Gym environment, making sequential decisions for maximizing cumulative rewards.

## SARSA LEARNING ALGORITHM
# Step1:
Initialize the Q-table with random values for all state-action pairs.

# Step 2:
Initialize the current state S and choose the initial action A using an epsilon-greedy policy based on the Q-values in the Q-table.

# Step 3:
Repeat until the episode ends and then take action A and observe the next state S' and the reward R.

# Step 4:
Update the Q-value for the current state-action pair (S, A) using the SARSA update rule.

# Step 5:
Update State and Action and repeat the step 3 untill the episodes ends.

## SARSA LEARNING FUNCTION
### Name:Daniel C
### Register Number:212223240023

```
import numpy as np

def sarsa(env,
          gamma=1.0,
          init_alpha=0.5,
          min_alpha=0.01,
          alpha_decay_ratio=0.5,
          init_epsilon=1.0,
          min_epsilon=0.1,
          epsilon_decay_ratio=0.9,
          n_episodes=3000):

    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)

    def decay_schedule(init_value, min_value, decay_ratio, n_episodes):
        decay_episodes = int(n_episodes * decay_ratio)
        values = np.linspace(init_value, min_value, decay_episodes)
        values = np.concatenate([values, np.full(n_episodes - decay_episodes, min_value)])
        return values

    alpha_schedule = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    epsilon_schedule = decay_schedule(init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes)

    for ep in range(n_episodes):
        alpha = alpha_schedule[ep]
        epsilon = epsilon_schedule[ep]
        result = env.reset()
        if isinstance(result, tuple):
            state = result[0]
        else:
            state = result
        done = False
        if np.random.rand() < epsilon:
            action = np.random.randint(nA)
        else:
            action = np.argmax(Q[state])
        while not done:
            step_result = env.step(action)
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, _ = step_result
            if np.random.rand() < epsilon:
                next_action = np.random.randint(nA)
            else:
                next_action = np.argmax(Q[next_state])
            td_target = reward + gamma * Q[next_state, next_action] * (not done)
            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error
            state, action = next_state, next_action
        Q_track[ep] = Q
        pi_track.append(np.argmax(Q, axis=1))
    pi = np.argmax(Q, axis=1)
    V = np.max(Q, axis=1)
    return Q, V, pi, Q_track, pi_track
```

## OUTPUT:
# Mention the optimal policy, optimal value function , success rate for the optimal policy.
<img width="806" height="1078" alt="image" src="https://github.com/user-attachments/assets/910d3bff-75b3-42f4-aa2f-c7ad792c6807" />


# state value functions of Monte Carlo method:
<img width="610" height="405" alt="image" src="https://github.com/user-attachments/assets/bc8830ad-c941-4fef-8e1a-58d383d1e7eb" />

<img width="1362" height="983" alt="image" src="https://github.com/user-attachments/assets/1df999a6-05c9-4293-9520-121131040fdc" />

# State value functions of SARSA learning:
<img width="642" height="400" alt="image" src="https://github.com/user-attachments/assets/75b35f87-be55-4642-8028-946afb311a23" />

<img width="1316" height="780" alt="image" src="https://github.com/user-attachments/assets/da014622-3bf3-48de-862b-170bcd411ced" />



## RESULT:
Thus to develop a Python program to find the optimal policy for the given RL environment using SARSA-Learning and compare the state values with the Monte Carlo method has been implemented successfully
