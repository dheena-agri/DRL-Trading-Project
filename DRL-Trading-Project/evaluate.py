import pandas as pd
from env.trading_env import TradingEnv
from utils.metrics import sharpe_ratio, max_drawdown

# Load rewards from training
rewards = []
with open("results/learning_curve.txt") as f:
    for line in f:
        rewards.append(float(line.strip().split(":")[1]))

# DRL Metrics
drl_sharpe = sharpe_ratio(rewards)
drl_drawdown = max_drawdown(rewards)

# Buy & Hold
data = pd.read_csv("data/spy.csv")
buy_hold_return = data["Close"].iloc[-1] / data["Close"].iloc[0] - 1

with open("results/performance_metrics.txt", "w") as f:
    f.write(f"DRL Sharpe Ratio: {drl_sharpe}\n")
    f.write(f"DRL Max Drawdown: {drl_drawdown}\n")
    f.write(f"Buy & Hold Return: {buy_hold_return}\n")

print("Evaluation complete. Metrics saved.")
