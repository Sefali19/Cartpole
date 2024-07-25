import gym
from reinforce import REINFORCE, REINFORCE_with_baseline
from plot_results import plot_results


def main():
    env = gym.make('CartPole-v1')

    reinforce_results = REINFORCE(env)
    reinforce_baseline_results = REINFORCE_with_baseline(env)

    plot_results(reinforce_results, reinforce_baseline_results)

    env.close()


if __name__ == "__main__":
    main()
