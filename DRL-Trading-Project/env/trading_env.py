import gymnasium as gym
import numpy as np


class TradingEnv(gym.Env):
    """
    A simple trading environment for a single stock.
    Actions:
        0 - Hold
        1 - Buy
        2 - Sell
    """

    def __init__(self, data, window_size=10):
        super().__init__()

        self.data = data.reset_index(drop=True)
        self.window_size = window_size

        self.current_step = window_size
        self.position = 0  # 0 = no position, 1 = holding
        self.cash = 1.0
        self.shares = 0.0

        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(window_size,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        self.current_step = self.window_size
        self.position = 0
        self.cash = 1.0
        self.shares = 0.0
        return self._get_state(), {}

    def _get_state(self):
        return self.data["Return"].iloc[
            self.current_step - self.window_size : self.current_step
        ].values.astype(np.float32)

    def step(self, action):
        price = self.data["Close"].iloc[self.current_step]
        prev_value = self._portfolio_value(price)

        # Execute action
        if action == 1 and self.position == 0:  # Buy
            self.shares = self.cash / price
            self.cash = 0.0
            self.position = 1

        elif action == 2 and self.position == 1:  # Sell
            self.cash = self.shares * price
            self.shares = 0.0
            self.position = 0

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1

        current_price = self.data["Close"].iloc[self.current_step]
        current_value = self._portfolio_value(current_price)

        reward = current_value - prev_value

        return self._get_state(), reward, done, False, {}

    def _portfolio_value(self, price):
        return self.cash + self.shares * price
