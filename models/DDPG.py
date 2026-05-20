# ═══════════════════════════════════════════════════════════
#  DDPG TRAIN FILE (VS CODE VERSION)
# ═══════════════════════════════════════════════════════════

# ==========================================================
# IMPORTS
# ==========================================================

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt

import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim

from collections import deque

# ✅ Import from Simulation.py
from Simulation import (
    RetailPricingEnv,
    verify_env_reward,
    plot_training,
    plot_price_behavior
)


# ==========================================================
# DEVICE
# ==========================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[DDPG] Using device: {DEVICE}")


# ==========================================================
# REPLAY BUFFER
# ==========================================================

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


# ==========================================================
# ACTOR NETWORK
# ==========================================================

class Actor(nn.Module):

    LOW = 0.7
    HIGH = 1.5

    def __init__(self, sd=7):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(sd, 256),
            nn.LayerNorm(256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        nn.init.xavier_uniform_(self.net[-2].weight)
        nn.init.zeros_(self.net[-2].bias)

    def forward(self, x):

        return self.LOW + (self.HIGH - self.LOW) * self.net(x)


# ==========================================================
# CRITIC NETWORK
# ==========================================================

class Critic(nn.Module):

    def __init__(self, sd=7):

        super().__init__()

        self.sb = nn.Sequential(

            nn.Linear(sd, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )

        self.co = nn.Sequential(

            nn.Linear(257, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 1)
        )

        nn.init.uniform_(self.co[-1].weight, -3e-3, 3e-3)
        nn.init.uniform_(self.co[-1].bias, -3e-3, 3e-3)

    def forward(self, s, a):

        return self.co(torch.cat([self.sb(s), a], dim=1))


# ==========================================================
# OU NOISE
# ==========================================================

class OUNoise:

    def __init__(
        self,
        mu=0,
        theta=0.15,
        sigma=0.3,
        dt=1e-2,
        smin=0.05,
        decay=0.9995
    ):

        self.mu = mu
        self.th = theta
        self.sigma = sigma
        self.smin = smin
        self.decay = decay
        self.dt = dt

        self.reset()

    def reset(self):

        self.x = 0.0

    def sample(self):

        self.x += (
            self.th * (self.mu - self.x) * self.dt
            + self.sigma * np.sqrt(self.dt) * np.random.randn()
        )

        return self.x

    def step_decay(self):

        self.sigma = max(self.sigma * self.decay, self.smin)


# ==========================================================
# DDPG AGENT
# ==========================================================

class DDPGAgent:

    def __init__(
        self,
        sd=7,
        alr=1e-4,
        clr=3e-4,
        gamma=0.99,
        tau=0.005,
        buf=100_000,
        bs=128
    ):

        self.gamma = gamma
        self.tau = tau
        self.bs = bs

        self.actor = Actor(sd).to(DEVICE)
        self.at = Actor(sd).to(DEVICE)

        self.critic = Critic(sd).to(DEVICE)
        self.ct = Critic(sd).to(DEVICE)

        self.at.load_state_dict(self.actor.state_dict())
        self.ct.load_state_dict(self.critic.state_dict())

        self.at.eval()
        self.ct.eval()

        self.ao = optim.Adam(
            self.actor.parameters(),
            lr=alr
        )

        self.co = optim.Adam(
            self.critic.parameters(),
            lr=clr,
            weight_decay=1e-4
        )

        self.buf = ReplayBuffer(buf)

        self.noise = OUNoise()

        self.al = []
        self.cl = []

    def act(self, state, noise=True):

        self.actor.eval()

        with torch.no_grad():

            a = self.actor(
                torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            ).item()

        self.actor.train()

        if noise:
            a = float(
                np.clip(
                    a + self.noise.sample(),
                    0.7,
                    1.5
                )
            )

        return float(a)

    def store(self, s, a, r, s2, d):

        self.buf.push(s, a, r, s2, d)

    def update(self):

        if len(self.buf) < self.bs:
            return None, None

        s, a, r, s2, d = self.buf.sample(self.bs)

        with torch.no_grad():

            y = r + self.gamma * (1 - d) * self.ct(
                s2,
                self.at(s2)
            )

        critic_loss = nn.MSELoss()(
            self.critic(s, a),
            y
        )

        self.co.zero_grad()

        critic_loss.backward()

        nn.utils.clip_grad_norm_(
            self.critic.parameters(),
            1.0
        )

        self.co.step()

        actor_loss = -self.critic(
            s,
            self.actor(s)
        ).mean()

        self.ao.zero_grad()

        actor_loss.backward()

        nn.utils.clip_grad_norm_(
            self.actor.parameters(),
            0.5
        )

        self.ao.step()

        for t, l in zip(
            self.at.parameters(),
            self.actor.parameters()v
        ):

            t.data.copy_(
                self.tau * l.data +
                (1 - self.tau) * t.data
            )

        for t, l in zip(
            self.ct.parameters(),
            self.critic.parameters()
        ):

            t.data.copy_(
                self.tau * l.data +
                (1 - self.tau) * t.data
            )

        self.noise.step_decay()

        self.cl.append(critic_loss.item())
        self.al.append(actor_loss.item())

        return critic_loss.item(), actor_loss.item()

    def save(self, path="ddpg_best.pth"):

        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }, path)

        print(f"  Model saved → {path}")


# ==========================================================
# TRAINING
# ==========================================================

def train_ddpg(
    env,
    agent,
    n_episodes=300,
    warmup=1000,
    verbose=25
):

    rewards = []
    prices = []

    best = -np.inf

    print(f"\n[Warmup] {warmup} random steps...")

    s = env.reset()

    for _ in range(warmup):

        a = random.uniform(0.7, 1.5)

        ns, r, done, _ = env.step(
            a,
            is_continuous=True
        )

        agent.store(s, a, r, ns, done)

        s = ns if not done else env.reset()

    print(f"[Warmup] Done. Buffer:{len(agent.buf)}")

    print(f"\n{'='*60}")
    print(f"  DDPG | {n_episodes} episodes")
    print(f"{'='*60}")

    for ep in range(1, n_episodes + 1):

        s = env.reset()

        agent.noise.reset()

        ep_r = 0
        ep_p = []

        while True:

            a = agent.act(s, noise=True)

            ns, r, done, info = env.step(
                a,
                is_continuous=True
            )

            agent.store(s, a, r, ns, done)

            agent.update()

            ep_r += r

            ep_p.append(info['price'])

            s = ns

            if done:
                break

        rewards.append(ep_r)

        prices.append(np.mean(ep_p))

        if ep_r > best:

            best = ep_r

            agent.save("ddpg_best.pth")

        if ep % verbose == 0 or ep == 1:

            print(
                f"  Ep{ep:>4}/{n_episodes} | "
                f"R:{ep_r:>8.1f} | "
                f"Avg25:{np.mean(rewards[-25:]):>8.1f} | "
                f"Price:${np.mean(ep_p):.2f} | "
                f"σ:{agent.noise.sigma:.4f}"
            )

    print(f"\n  Done! Best:{best:.1f}")

    return {
        'episode_rewards': rewards,
        'avg_prices': prices,
        'actor_losses': agent.al,
        'critic_losses': agent.cl
    }


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    verify_env_reward()

    env = RetailPricingEnv(seed=42)

    agent = DDPGAgent(sd=env.state_dim)

    print(
        f"\nActor : "
        f"{sum(p.numel() for p in agent.actor.parameters()):,} params"
    )

    print(
        f"Critic: "
        f"{sum(p.numel() for p in agent.critic.parameters()):,} params"
    )

    hist = train_ddpg(
        env,
        agent,
        n_episodes=300,
        warmup=1000,
        verbose=25
    )

    print("\nGenerating graphs...")

    plot_training(hist)

    plot_price_behavior(
        RetailPricingEnv(seed=7),
        agent
    )

    print("\nTraining Complete ✅")