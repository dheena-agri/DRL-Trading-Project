# Deep Reinforcement Learning for Algorithmic Trading

## Overview
This project implements a **Deep Reinforcement Learning (DRL)** agent for algorithmic trading using historical financial time series data.  
Instead of predicting future prices, the trading problem is framed as a **Markov Decision Process (MDP)** where an agent learns optimal trading decisions through interaction with a simulated market environment.

A **Deep Q-Network (DQN)** is implemented **from scratch** (without high-level RL wrappers) and evaluated against a **Buy-and-Hold benchmark** using risk-adjusted performance metrics.

---

## Problem Statement
Financial markets involve sequential decision-making under uncertainty. Traditional supervised learning approaches fail to capture the dynamic feedback between actions and future states.

This project addresses the problem by:
- Modeling trading as an MDP
- Training a DRL agent to decide **Buy / Sell / Hold**
- Comparing learned policies with a standard benchmark strategy

---

## Markov Decision Process (MDP) Formulation

### State Space
- Last **10 daily returns** of the asset  
- Captures short-term market dynamics while remaining computationally simple

### Action Space
| Action | Description |
|------|-------------|
| 0 | Hold |
| 1 | Buy |
| 2 | Sell |

Only one position can be held at a time.

### Reward Function
Reward is defined as the **change in portfolio value** between two consecutive timesteps:

\[
R_t = V_{t+1} - V_t
\]

This directly aligns the learning objective with profit maximization.

### Episode Termination
An episode ends when the agent reaches the final timestep of the historical dataset.

---

## Data
- Asset: **S&P 500 ETF (SPY)**
- Frequency: Daily
- Source: Yahoo Finance
- Stored locally as `data/spy.csv` for reproducibility

---

## Algorithm
### Deep Q-Network (DQN)

The DQN approximates the optimal action-value function \( Q(s,a) \) using a neural network.

**Network Architecture**
- Input layer: 10 neurons (state size)
- Hidden layers: 2 layers × 64 neurons (ReLU)
- Output layer: 3 neurons (actions)

**Key Techniques**
- Experience Replay Buffer
- Epsilon-Greedy Exploration
- Mean Squared Error (MSE) loss

---

## Hyperparameters

| Parameter | Value |
|--------|------|
| Learning Rate | 0.001 |
| Discount Factor (γ) | 0.99 |
| Batch Size | 32 |
| Episodes | 200 |
| Epsilon | 0.1 |

These values were selected empirically to ensure stable training.

---

## Training and Learning Curve
During training:
- Early episodes show high reward volatility due to exploration
- Reward variance decreases over time
- Policy converges to stable behavior

**Textual Learning Curve Summary**
- Episodes 0–30: Highly unstable rewards
- Episodes 30–100: Reduced volatility
- Episodes 100–200: Stable convergence

---

## Benchmark Strategy
A **Buy-and-Hold** strategy is used as a baseline:
- Asset purchased at the beginning
- Held until the end of the test period

---

## Performance Metrics
The DRL agent is evaluated using:

- **Sharpe Ratio** (risk-adjusted return)
- **Maximum Drawdown** (downside risk)

Results are saved in:
