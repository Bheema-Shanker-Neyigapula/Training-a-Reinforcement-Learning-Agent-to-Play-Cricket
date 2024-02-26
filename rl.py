import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Define a simple Q-network
class QNetwork(nn.Module):
    def __init__(self, input_dim, num_actions):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# Define the cricket environment
class CricketEnv(gym.Env):
    def __init__(self):
        # Define action and observation space
        self.action_space = gym.spaces.Discrete(3)  # 3 actions: bowl, bat, defend
        self.observation_space = gym.spaces.Box(low=0, high=100, shape=(3,), dtype=np.float32)  # 3 observations: runs, wickets, overs

        # Initialize state variables
        self.runs = 0
        self.wickets = 0
        self.overs = 0

    def reset(self):
        # Reset state variables
        self.runs = 0
        self.wickets = 0
        self.overs = 0
        return self._get_observation()

    def step(self, action):
        # Execute action and update state variables
        if action == 0:  # bowl
            self.runs += np.random.randint(0, 7)  # Simulate runs scored
            self.wickets += np.random.choice([0, 1], p=[0.8, 0.2])  # Simulate wicket probability
        elif action == 1:  # bat
            self.runs += np.random.randint(0, 10)  # Simulate runs scored
        elif action == 2:  # defend
            pass  # No changes in state for defending

        # Update overs
        if np.random.rand() < 0.1:  # 10% chance of completing an over
            self.overs += 1

        # Define rewards and termination conditions
        reward = self.runs + (self.wickets * -20) - self.overs
        done = self.overs >= 10  # Terminate after 10 overs

        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        return np.array([self.runs, self.wickets, self.overs], dtype=np.float32)

# Define DQN Agent
class DQNAgent:
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
        self.q_network = QNetwork(observation_space.shape[0], action_space.n)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-4)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.replay_buffer = ReplayBuffer(capacity=10000)

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return self.action_space.sample()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state)
                q_values = self.q_network(state)
                return q_values.argmax().item()

    def train(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return
        transitions = self.replay_buffer.sample(batch_size)
        batch = list(zip(*transitions))
        state_batch = torch.FloatTensor(batch[0])
        action_batch = torch.LongTensor(batch[1])
        reward_batch = torch.FloatTensor(batch[2])
        next_state_batch = torch.FloatTensor(batch[3])
        done_batch = torch.BoolTensor(batch[4])

        current_q_values = self.q_network(state_batch).gather(1, action_batch.unsqueeze(1))
        next_q_values = self.q_network(next_state_batch).max(1)[0].detach()
        expected_q_values = reward_batch + (1 - done_batch.float()) * self.gamma * next_q_values

        loss = F.smooth_l1_loss(current_q_values.squeeze(), expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# Initialize cricket environment and DQN agent
env = CricketEnv()
agent = DQNAgent(env.observation_space, env.action_space)

# Lists to store rewards and epsilon values for plotting learning curves
all_rewards = []
all_epsilons = []

# Train the agent
num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
    agent.train(batch_size=32)

    # Store rewards and epsilon values for plotting
    all_rewards.append(total_reward)
    all_epsilons.append(agent.epsilon)

    # Print episode information every 10 episodes
    if (episode + 1) % 10 == 0:
        print(f"Episode {episode + 1}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}")

# Plotting learning curves
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(all_rewards, label='Total Reward')
plt.title('Total Reward Over Episodes')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(all_epsilons, label='Epsilon')
plt.title('Epsilon Decay Over Episodes')
plt.xlabel('Episode')
plt.ylabel('Epsilon')
plt.legend()

plt.tight_layout()
plt.show()

# Save the trained model
torch.save(agent.q_network.state_dict(), "dqn_cricket_model.pth")
