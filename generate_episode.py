import numpy as np

def generate_episode(env, theta, policy, display=False):
    """ Generates one episode and returns the list of states, the list of rewards, and the list of actions of that episode """
    state = env.reset()
    states = [state]
    actions = []
    rewards = []
    for t in range(500):
        if display:
            env.render()
        p = policy(state, theta)
        action = np.random.choice(len(p), p=p)
        state, reward, done, _ = env.step(action)
        rewards.append(reward)
        actions.append(action)
        if done:
            break
        states.append(state)
    return states, rewards, actions
