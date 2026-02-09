import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np

from env.trading_env import TradingEnv
from models.dqn import DQN
from utils.replay_buffer import ReplayBuffer

# -----------------------------
# Hyperparameters
# -----------------------------
WINDOW_SIZE = 10
EPISODES = 200
BATCH_SIZE = 32
GAMMA = 0.99
LR = 1e-3
EPSILON = 0.1

# -----------------------------
# Load Data
# -----------------------------
data = pd.read_csv("data/spy.csv")
env = TradingEnv(data, WINDOW_SIZE)

# -----------------------------
# Model Setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DQN(WINDOW_SIZE, 3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
buffer = ReplayBuffer()

episode_rewards = []

# -----------------------------
# Training Loop
# -----------------------------
for episode in range(EPISODES):
    state, _ = env.reset()
    total_reward = 0

    while True:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

        # Epsilon-greedy
        if np.random.rand() < EPSILON:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = torch.argmax(model(state_tensor)).item()

        next_state, reward, done, _, _ = env.step(action)
        buffer.push(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward

        if len(buffer) >= BATCH_SIZE:
            states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)

            states = torch.FloatTensor(states).to(device)
            actions = torch.LongTensor(actions).to(device)
            rewards = torch.FloatTensor(rewards).to(device)
            next_states = torch.FloatTensor(next_states).to(device)
            dones = torch.FloatTensor(dones).to(device)

            q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze()
            next_q_values = model(next_states).max(1)[0]
            target = rewards + GAMMA * next_q_values * (1 - dones)

            loss = F.mse_loss(q_values, target.detach())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done:
            break

    episode_rewards.append(total_reward)

    if episode % 20 == 0:
        print(f"Episode {episode}, Reward: {total_reward:.4f}")

# Save learning curve
with open("results/learning_curve.txt", "w") as f:
    for i, r in enumerate(episode_rewards):
        f.write(f"Episode {i}: {r}\n")
