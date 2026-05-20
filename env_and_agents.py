import numpy as np
import random
from collections import deque
import json

# ─── ENVIRONMENT ───────────────────────────────────────────────────────────────
class RetailPricingEnv:
    def __init__(self, seed=None):
        self.max_inventory = 100
        self.base_price = 100.0
        self.unit_cost = 60.0
        self.holding_cost_per_unit = 0.5
        self.stockout_penalty_val = 25.0
        self.max_steps = 720
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
            multiplier = float(np.clip(action, 0.7, 1.5))
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
        info = {"price": self.current_price, "demand": round(demand, 2),
                "sold": units_sold, "inventory": self.inventory,
                "product_age_pct": round((self.step_count / self.max_steps) * 100, 1),
                "competitor_price": round(self.competitor_price, 2)}
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
        return np.array([
            self.inventory / self.max_inventory,
            (self.current_price - 70) / 80.0,
            self.hour / 23.0,
            self.day / 6.0,
            np.mean(self.demand_history) / 25.0,
            (self.competitor_price - 70) / 80.0,
            self.step_count / self.max_steps
        ], dtype=np.float32)

# ─── Q-LEARNING ────────────────────────────────────────────────────────────────
class QLearningAgent:
    def __init__(self, n_actions=5, alpha=0.1, gamma=0.95,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.bins = [
            np.linspace(0, 1, 5), np.linspace(0, 1, 5),
            np.linspace(0, 1, 4), np.linspace(0, 1, 3),
            np.linspace(0, 1, 4), np.linspace(0, 1, 4),
            np.linspace(0, 1, 4),
        ]
        q_shape = tuple(len(b) + 1 for b in self.bins) + (n_actions,)
        self.q_table = np.zeros(q_shape)

    def _discretize(self, obs):
        return tuple(np.digitize(obs[i], self.bins[i]) for i in range(len(self.bins)))

    def select_action(self, obs, greedy=False):
        if not greedy and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        return int(np.argmax(self.q_table[self._discretize(obs)]))

    def update(self, obs, action, reward, next_obs, done):
        s = self._discretize(obs)
        s2 = self._discretize(next_obs)
        current_q = self.q_table[s][action]
        max_next_q = 0 if done else np.max(self.q_table[s2])
        self.q_table[s][action] += self.alpha * (reward + self.gamma * max_next_q - current_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path):
        np.savez(
            path,
            q_table=self.q_table,
            bins=np.array(self.bins, dtype=object),
            epsilon=self.epsilon,
            alpha=self.alpha,
            gamma=self.gamma,
            epsilon_end=self.epsilon_end,
            epsilon_decay=self.epsilon_decay,
        )

    @classmethod
    def load(cls, path):
        data = np.load(path, allow_pickle=True)
        agent = cls(
            alpha=float(data["alpha"]),
            gamma=float(data["gamma"]),
            epsilon_end=float(data["epsilon_end"]),
            epsilon_decay=float(data["epsilon_decay"]),
        )
        agent.q_table = data["q_table"]
        agent.bins = [np.asarray(b) for b in data["bins"]]
        agent.epsilon = float(data["epsilon"])
        return agent

# ─── BASELINES ─────────────────────────────────────────────────────────────────
class FixedAgent:
    def select_action(self, obs): return 2  # always 0%

class RuleAgent:
    def select_action(self, obs):
        inv = obs[0]; hour = int(obs[2]*23)
        our_p = obs[1]; comp_p = obs[5]
        peak = (10<=hour<=13) or (18<=hour<=21)
        if inv < 0.2: return 4
        if inv > 0.7 and not peak: return 0
        if peak: return 3
        if comp_p < our_p - 0.05: return 1
        return 2

# ─── DDPG (PyTorch) ───────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.optim as optim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self, cap=100_000):
        self.buf = deque(maxlen=cap)

    def push(self, s, a, r, s2, d):
        self.buf.append((
            np.array(s, dtype=np.float32),
            np.array([a], dtype=np.float32),
            np.array([r], dtype=np.float32),
            np.array(s2, dtype=np.float32),
            np.array([d], dtype=np.float32)
        ))

    def sample(self, bs):
        batch = random.sample(self.buf, bs)
        s, a, r, s2, d = zip(*batch)
        f = lambda x: torch.FloatTensor(np.stack(x)).to(DEVICE)
        return f(s), f(a), f(r), f(s2), f(d)

    def __len__(self):
        return len(self.buf)


class Actor(nn.Module):
    LOW = 0.7
    HIGH = 1.5

    def __init__(self, sd=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(sd, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Sigmoid()
        )
        nn.init.xavier_uniform_(self.net[-2].weight)
        nn.init.zeros_(self.net[-2].bias)

    def forward(self, x):
        return self.LOW + (self.HIGH - self.LOW) * self.net(x)


class Critic(nn.Module):
    def __init__(self, sd=7):
        super().__init__()
        self.sb = nn.Sequential(nn.Linear(sd, 256), nn.LayerNorm(256), nn.ReLU())
        self.co = nn.Sequential(
            nn.Linear(257, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
        nn.init.uniform_(self.co[-1].weight, -3e-3, 3e-3)
        nn.init.uniform_(self.co[-1].bias, -3e-3, 3e-3)

    def forward(self, s, a):
        return self.co(torch.cat([self.sb(s), a], dim=1))


class OUNoise:
    def __init__(self, mu=0, theta=0.15, sigma=0.3, dt=1e-2, smin=0.05, decay=0.9995):
        self.mu = mu; self.th = theta; self.sigma = sigma
        self.smin = smin; self.decay = decay; self.dt = dt
        self.reset()

    def reset(self):
        self.x = 0.0

    def sample(self):
        self.x += self.th * (self.mu - self.x) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.randn()
        return self.x

    def step_decay(self):
        self.sigma = max(self.sigma * self.decay, self.smin)


class DDPGAgent:
    def __init__(self, sdim=7, alo=0.7, ahi=1.5, gamma=0.99, tau=0.005,
                 alr=1e-4, clr=3e-4, bs=128, buf=100_000):
        self.gamma = gamma; self.tau = tau; self.bs = bs

        self.actor  = Actor(sdim).to(DEVICE)
        self.at     = Actor(sdim).to(DEVICE)
        self.critic = Critic(sdim).to(DEVICE)
        self.ct     = Critic(sdim).to(DEVICE)

        self.at.load_state_dict(self.actor.state_dict())
        self.ct.load_state_dict(self.critic.state_dict())
        self.at.eval(); self.ct.eval()

        self.ao = optim.Adam(self.actor.parameters(), lr=alr)
        self.co = optim.Adam(self.critic.parameters(), lr=clr, weight_decay=1e-4)

        self.buf   = ReplayBuffer(buf)
        self.noise = OUNoise()
        self.al = []; self.cl = []

    def act(self, state, noise=True):
        self.actor.eval()
        with torch.no_grad():
            a = self.actor(torch.FloatTensor(state).unsqueeze(0).to(DEVICE)).item()
        self.actor.train()
        if noise:
            a = float(np.clip(a + self.noise.sample(), 0.7, 1.5))
        return float(a)

    def store(self, s, a, r, s2, d):
        self.buf.push(s, a, r, s2, d)

    def update(self):
        if len(self.buf) < self.bs:
            return None, None
        s, a, r, s2, d = self.buf.sample(self.bs)
        with torch.no_grad():
            y = r + self.gamma * (1 - d) * self.ct(s2, self.at(s2))
        critic_loss = nn.MSELoss()(self.critic(s, a), y)
        self.co.zero_grad(); critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.co.step()
        actor_loss = -self.critic(s, self.actor(s)).mean()
        self.ao.zero_grad(); actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.ao.step()
        for t, l in zip(self.at.parameters(), self.actor.parameters()):
            t.data.copy_(self.tau * l.data + (1 - self.tau) * t.data)
        for t, l in zip(self.ct.parameters(), self.critic.parameters()):
            t.data.copy_(self.tau * l.data + (1 - self.tau) * t.data)
        self.noise.step_decay()
        self.cl.append(critic_loss.item()); self.al.append(actor_loss.item())
        return critic_loss.item(), actor_loss.item()

    # Legacy alias so existing callers of train_step() still work
    def train_step(self):
        return self.update()

    def save(self, path="ddpg_best.pth"):
        torch.save({'actor': self.actor.state_dict(), 'critic': self.critic.state_dict()}, path)
        print(f"  Model saved → {path}")


    @classmethod
    def load(cls, path="ddpg_best.pth", sdim=7):
        agent = cls(sdim=sdim)
        checkpoint = torch.load(path, map_location=DEVICE)
        agent.actor.load_state_dict(checkpoint['actor'])
        agent.critic.load_state_dict(checkpoint['critic'])
        agent.at.load_state_dict(checkpoint['actor'])
        agent.ct.load_state_dict(checkpoint['critic'])
        agent.actor.eval()
        agent.at.eval()
        agent.critic.eval()
        agent.ct.eval()
        print(f"  Loaded DDPG model from {path}")
        return agent
