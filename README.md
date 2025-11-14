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
<img width="590" height="722" alt="image" src="https://github.com/user-attachments/assets/dd625b58-1c4e-4072-b1b3-0024b1be6419" />


# state value functions of Monte Carlo method:
<img width="472" height="258" alt="image" src="https://github.com/user-attachments/assets/2951edf2-e6e1-4db5-8283-32169e4e85b6" />

<img width="758" height="647" alt="image" src="https://github.com/user-attachments/assets/bf08f60f-356b-463a-8791-4c781c7068cb" />


# State value functions of SARSA learning:
<img width="437" height="261" alt="image" src="https://github.com/user-attachments/assets/b3d75e80-0d05-4e4c-b6f9-8b8683382699" />

<img width="746" height="645" alt="image" src="https://github.com/user-attachments/assets/7c2c03d0-b8df-4222-a325-e676925365a1" />

## Graph comparison:
<img width="1692" height="615" alt="image" src="https://github.com/user-attachments/assets/024ac68d-4991-471f-a91e-2cb85f547cd0" />

<img width="1759" height="612" alt="image" src="https://github.com/user-attachments/assets/d618bbde-ed9f-4885-a08c-dc03c41bd3bc" />



## RESULT:
Thus to develop a Python program to find the optimal policy for the given RL environment using SARSA-Learning and compare the state values with the Monte Carlo method has been implemented successfully
