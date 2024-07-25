import matplotlib.pyplot as plt

def plot_results(reinforce_results, reinforce_baseline_results):
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(reinforce_results)), reinforce_results, label='REINFORCE')
    plt.plot(range(len(reinforce_baseline_results)), reinforce_baseline_results, label='REINFORCE with Baseline')
    plt.xlabel('Episode')
    plt.ylabel('Mean Episode Length (last 100 episodes)')
    plt.title('REINFORCE vs REINFORCE with Baseline on CartPole-v1')
    plt.legend()
    plt.show()
