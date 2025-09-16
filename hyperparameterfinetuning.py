from agent_setup import run_dqn
import matplotlib.pyplot as plt
import numpy as np

episode_rewards, epsilon_values = run_dqn()
 
def plot_epsilonvsrewards(episode_rewards, epsilon_values):
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward', color=color)
    ax1.plot(episode_rewards, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:red'
    ax2.set_ylabel('Epsilon', color=color)  # we already handled the x-label with ax1
    ax2.plot(epsilon_values, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title('Epsilon vs Total Reward per Episode')
    plt.show()

def plot_rewards(episode_rewards):
    def moving_average(x, window=50):
        return np.convolve(x, np.ones(window)/window, mode="valid")

    plt.figure(figsize=(10,5))
    plt.plot(episode_rewards, label="Episode Reward", alpha=0.3)
    plt.plot(moving_average(episode_rewards), label="Moving Average (50)", color="red")
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Learning Curve")
    plt.legend()
    plt.show()

plot_rewards(episode_rewards)