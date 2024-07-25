import numpy as np

def policy(state, theta):
    """ Return probabilities for actions under softmax action selection """
    h = np.dot(state, theta)
    exp_h = np.exp(h - np.max(h))  # Subtract max for numerical stability
    return exp_h / np.sum(exp_h)
