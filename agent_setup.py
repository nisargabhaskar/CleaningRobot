import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque
import numpy as np
import gymnasium as gym
from env_setup import GridWorldEnv

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class ReplayBuffer:
    def __init__(self, capacity=25000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


# Init env + models
def run_dqn(gamma = 0.99, batch_size = 128, lr = 1e-4, epsilon = 1.0, epsilon_min = 0.05, epsilon_decay = 0.995):
    gym.register(
        id="gymnasium_env/GridWorld-v0",
        entry_point=GridWorldEnv,
        max_episode_steps=600,  # Prevent infinite episodes
    )
    env = gym.make("gymnasium_env/GridWorld-v0")
    state_dim = 2          # since agent pos is (x, y)
    action_dim = env.action_space.n

    q_net = QNetwork(state_dim, action_dim)
    target_net = QNetwork(state_dim, action_dim)
    target_net.load_state_dict(q_net.state_dict())   # sync weights

    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    replay_buffer = ReplayBuffer()

    num_episodes = 500
    sync_target_steps = 100   # how often to update target net

    episode_rewards = []
    epsilon_values = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = state["agent"]   # only use agent coords for now
        state = np.array(state, dtype=np.float32)

        total_reward = 0

        done = False
        while not done:
            # Epsilon-greedy action
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = q_net(torch.tensor(state).float().unsqueeze(0)).argmax().item()

            # Step environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = np.array(next_state["agent"], dtype=np.float32)

            done = terminated or truncated
            total_reward += reward

            # Store in replay buffer
            replay_buffer.push(state, action, reward, next_state, done)

            state = next_state

            # Training step
            if len(replay_buffer) >= batch_size:
                s, a, r, s2, d = replay_buffer.sample(batch_size)

                s = torch.tensor(s, dtype=torch.float32)
                a = torch.tensor(a, dtype=torch.int64).unsqueeze(1)
                r = torch.tensor(r, dtype=torch.float32).unsqueeze(1)
                s2 = torch.tensor(s2, dtype=torch.float32)
                d = torch.tensor(d, dtype=torch.float32).unsqueeze(1)

                # Q(s,a)
                q_values = q_net(s).gather(1, a)

                # Bellman target
                with torch.no_grad():
                    max_next_q = target_net(s2).max(1)[0].unsqueeze(1)
                    target = r + gamma * max_next_q * (1 - d)

                # Loss
                loss = F.mse_loss(q_values, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Decay epsilon
        # epsilon = max(epsilon_min, epsilon * epsilon_decay)
        epsilon = max(epsilon_min, 1 - episode/500)

        # Sync target net
        if episode % sync_target_steps == 0:
            target_net.load_state_dict(q_net.state_dict())
        episode_rewards.append(total_reward)
        epsilon_values.append(epsilon)
        print(f"Episode {episode}, Reward: {total_reward}")
    return episode_rewards, epsilon_values 