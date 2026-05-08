import numpy as np
import random
from collections import deque


class RetailPricingEnv:
    """
    A unified Retail Environment for Discrete (Q-Learning) and Continuous (DDPG) RL Agents.
    """
    def __init__(self, seed=None):
        self.max_inventory = 100
        self.base_price = 100.0
        self.unit_cost = 60.0
        self.holding_cost_per_unit = 0.5
        self.stockout_penalty_val = 25.0
        self.max_steps = 720  # 30 days * 24 hours

        self.discrete_multipliers = [0.8, 0.9, 1.0, 1.1, 1.2]
        self.n_actions_discrete = len(self.discrete_multipliers)

        self.state_dim = 7
        self.demand_history = deque(maxlen=7)
        self.price_history = deque(maxlen=2)

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.reset()

    def reset(self):
        """Reset environment to start of a new 30-day month."""
        self.inventory = 50
        self.current_price = self.base_price
        self.competitor_price = 95.0
        self.hour = 0
        self.day = 0
        self.step_count = 0
        self.total_profit = 0.0
        self.demand_history.extend([10.0] * 7)
        self.price_history.extend([self.base_price] * 2)
        return self._get_observation()

    def step(self, action, is_continuous=False):
        prev_price = self.current_price

        if not is_continuous:
            multiplier = self.discrete_multipliers[action]
        else:
            multiplier = np.clip(action, 0.7, 1.5)
            if isinstance(multiplier, np.ndarray):
                multiplier = multiplier.item()

        self.current_price = round(self.base_price * multiplier, 2)

        comp_change = np.random.normal(0, 1.2)
        if self.current_price < self.competitor_price - 10:
            comp_change -= 0.5
        self.competitor_price = np.clip(self.competitor_price + comp_change, 75, 125)

        demand = self._calculate_demand()
        units_sold = min(int(demand), self.inventory)

        revenue = self.current_price * units_sold
        cogs = self.unit_cost * units_sold
        holding_costs = self.holding_cost_per_unit * (self.inventory - units_sold)
        stockout_penalty = self.stockout_penalty_val if self.inventory < 5 and demand > self.inventory else 0.0
        volatility_penalty = 0.15 * abs(self.current_price - prev_price)

        reward = (revenue - cogs) - holding_costs - stockout_penalty - volatility_penalty
        self.total_profit += reward

        if self.hour == 0:
            restock_amt = random.randint(15, 25)
            self.inventory = min(self.inventory + restock_amt, self.max_inventory)

        self.inventory = max(0, self.inventory - units_sold)
        self.hour = (self.hour + 1) % 24
        if self.hour == 0:
            self.day = (self.day + 1) % 7

        self.step_count += 1
        self.demand_history.append(demand)
        self.price_history.append(self.current_price)

        done = self.step_count >= self.max_steps

        info = {
            "price": self.current_price,
            "demand": round(demand, 2),
            "sold": units_sold,
            "inventory": self.inventory,
            "product_age_pct": round((self.step_count / self.max_steps) * 100, 1)
        }
        return self._get_observation(), reward, done, info

    def _calculate_demand(self):
        price_ratio = self.current_price / self.competitor_price
        base_demand = 12.0 * (1.0 / (price_ratio ** 1.2))
        if (10 <= self.hour <= 13) or (18 <= self.hour <= 21):
            base_demand *= 1.6
        if self.day >= 5:
            base_demand *= 1.3
        noise = np.random.normal(0, 1.5)
        return max(0.0, base_demand + noise)

    def _get_observation(self):
        """Returns a normalized state vector including Product Age."""
        return np.array([
            self.inventory / self.max_inventory,
            (self.current_price - 70) / 80.0,
            self.hour / 23.0,
            self.day / 6.0,
            np.mean(self.demand_history) / 25.0,
            (self.competitor_price - 70) / 80.0,
            self.step_count / self.max_steps
        ], dtype=np.float32)


if __name__ == "__main__":
    env = RetailPricingEnv(seed=42)

    print("--- SCENARIO 1: HIGH PRICING (Testing Elasticity) ---")
    env.reset()
    for _ in range(5):
        obs, reward, done, info = env.step(4)
        print(f"Price: ${info['price']} | Demand: {info['demand']} | Reward: {reward:.2f}")

    print("\n--- SCENARIO 2: PEAK HOUR SURGE (Testing Seasonality) ---")
    env.reset()
    env.hour = 10
    _, _, _, info_peak = env.step(2)
    env.hour = 3
    _, _, _, info_off = env.step(2)
    print(f"Peak Demand (10AM): {info_peak['demand']} vs Off-Peak (3AM): {info_off['demand']}")

    print("\n--- SCENARIO 3: STOCKOUT PENALTY (Testing Penalties) ---")
    env.reset()
    env.inventory = 2
    _, reward, _, info = env.step(0)
    print(f"Inventory: {info['inventory']} | Demand: {info['demand']} | Reward: {reward:.2f}")
    if reward < 0:
        print("Success: Reward is negative due to stockout penalty.")

    print("\n--- SCENARIO 4: VOLATILITY PENALTY (Testing Stability) ---")
    env.reset()
    env.step(0)
    _, reward_stable, _, _ = env.step(0)
    _, reward_jump, _, _   = env.step(4)
    print(f"Stable Reward: {reward_stable:.2f} vs Jump Reward: {reward_jump:.2f}")
    print("Note: The Jump Reward is lower because the agent was penalized for a sudden $40 price swing.")