import numpy as np
from generate_episode import generate_episode
from policy import policy

def REINFORCE(env):
    theta = np.random.rand(4, 2)  # policy parameters
    alpha = 0.01  # learning rate
    gamma = 0.99  # discount factor
    episode_lengths = []
    mean_lengths = []

    for e in range(10000):
        if e % 300 == 0:
            states, rewards, actions = generate_episode(env, theta, policy, True)  # display the policy every 300 episodes
        else:
            states, rewards, actions = generate_episode(env, theta, policy, False)

        episode_length = len(states)
        episode_lengths.append(episode_length)
        if len(episode_lengths) > 100:
            episode_lengths.pop(0)
        mean_length = np.mean(episode_lengths)
        mean_lengths.append(mean_length)

        print(f"Episode: {e}, Length: {episode_length}, Mean Length: {mean_length:.2f}")

        # Implement the REINFORCE algorithm
        G = 0
        for t in range(len(states) - 1, -1, -1):
            G = gamma * G + rewards[t]
            state = states[t]
            action = actions[t]

            # Compute gradient
            probs = policy(state, theta)
            grad = np.outer(state, probs)
            grad[:, action] -= state

            # Update policy parameters
            theta += alpha * G * grad

        if mean_length >= 495:
            print(f"Solved in {e} episodes!")
            break

    return mean_lengths

def REINFORCE_with_baseline(env):
    theta = np.random.rand(4, 2)  # policy parameters
    w = np.random.rand(4)  # baseline parameters
    alpha = 0.01  # learning rate for policy
    beta = 0.01  # learning rate for baseline
    gamma = 0.99  # discount factor
    episode_lengths = []
    mean_lengths = []

    for e in range(10000):
        if e % 300 == 0:
            states, rewards, actions = generate_episode(env, theta, policy, True)  # display the policy every 300 episodes
        else:
            states, rewards, actions = generate_episode(env, theta, policy, False)

        episode_length = len(states)
        episode_lengths.append(episode_length)
        if len(episode_lengths) > 100:
            episode_lengths.pop(0)
        mean_length = np.mean(episode_lengths)
        mean_lengths.append(mean_length)

        print(f"Episode: {e}, Length: {episode_length}, Mean Length: {mean_length:.2f}")

        # Implement the REINFORCE with baseline algorithm
        G = 0
        for t in range(len(states) - 1, -1, -1):
            G = gamma * G + rewards[t]
            state = states[t]
            action = actions[t]

            # Calculate baseline value
            V = np.dot(w, state)

            # Compute gradients
            probs = policy(state, theta)
            policy_grad = np.outer(state, probs)
            policy_grad[:, action] -= state

            # Update policy parameters
            theta += alpha * (G - V) * policy_grad

            # Update baseline parameters
            w += beta * (G - V) * state

        if mean_length >= 495:
            print(f"Solved in {e} episodes!")
            break

    return mean_lengths
