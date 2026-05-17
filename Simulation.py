# ═══════════════════════════════════════════════════════════
#  CELL 1 — ENVIRONMENT + VERIFY + PLOTS
# ═══════════════════════════════════════════════════════════

import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════════════
#  ENVIRONMENT
# ═══════════════════════════════════════════════════════════
class RetailPricingEnv:
    def __init__(self, seed=None):
        self.max_inventory         = 100
        self.base_price            = 100.0
        self.unit_cost             = 75.0
        self.holding_cost_per_unit = 0.8
        self.stockout_penalty_val  = 30.0
        self.max_steps             = 720

        self.discrete_multipliers  = [0.8, 0.9, 1.0, 1.1, 1.2]
        self.n_actions_discrete    = len(self.discrete_multipliers)
        self.state_dim             = 7

        self.demand_history = deque(maxlen=7)
        self.price_history  = deque(maxlen=2)

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.reset()

    def reset(self):
        self.inventory        = 50
        self.current_price    = self.base_price
        self.competitor_price = 95.0
        self.hour             = 0
        self.day              = 0
        self.step_count       = 0
        self.total_profit     = 0.0
        self.demand_history.extend([20.0] * 7)
        self.price_history.extend([self.base_price] * 2)
        return self._get_observation()

    def _calculate_demand(self):
        price = self.current_price
        comp  = self.competitor_price

        threshold  = comp * 1.0
        k          = 0.15
        max_demand = 65.0

        base_demand = max_demand / (1.0 + np.exp(k * (price - threshold)))

        if price > comp * 1.15:
            base_demand *= 0.05

        if (10 <= self.hour <= 13) or (18 <= self.hour <= 21):
            base_demand *= 1.5
        if self.day >= 5:
            base_demand *= 1.25

        return max(0.0, base_demand + np.random.normal(0, 1.5))

    def step(self, action, is_continuous=False):
        prev_price = self.current_price

        if not is_continuous:
            multiplier = self.discrete_multipliers[int(action)]
        else:
            multiplier = float(np.clip(action, 0.7, 1.5))

        self.current_price = round(self.base_price * multiplier, 2)

        comp_change = np.random.normal(0, 1.0)
        if self.current_price < self.competitor_price - 10:
            comp_change -= 0.8
        elif self.current_price > self.competitor_price + 15:
            comp_change += 1.2

        # ✅ FIX 1: Competitor range tightened 75-125 → 85-105
        self.competitor_price = float(np.clip(
            self.competitor_price + comp_change, 85, 105))

        demand     = self._calculate_demand()
        units_sold = min(int(demand), self.inventory)
        unsold     = max(0, self.inventory - units_sold)

        revenue  = self.current_price * units_sold
        cogs     = self.unit_cost * units_sold
        holding  = self.holding_cost_per_unit * unsold
        stockout = self.stockout_penalty_val if (
            self.inventory < 5 and demand > self.inventory) else 0.0
        vol_pen  = 0.15 * abs(self.current_price - prev_price)

        reward = (revenue - cogs) - holding - stockout - vol_pen
        reward = reward / 100.0

        self.total_profit += reward

        if self.hour == 0:
            restock = random.randint(15, 25)
            self.inventory = min(self.inventory + restock, self.max_inventory)

        self.inventory  = max(0, self.inventory - units_sold)
        self.hour       = (self.hour + 1) % 24
        if self.hour == 0:
            self.day = (self.day + 1) % 7
        self.step_count += 1
        self.demand_history.append(demand)
        self.price_history.append(self.current_price)

        done = self.step_count >= self.max_steps
        info = {
            "price":     self.current_price,
            "demand":    round(demand, 2),
            "sold":      units_sold,
            "inventory": self.inventory,
            "product_age_pct": round(self.step_count / self.max_steps * 100, 1),
        }
        return self._get_observation(), reward, done, info

    def _get_observation(self):
        return np.array([
            self.inventory               / self.max_inventory,
            (self.current_price  - 70)   / 80.0,
            self.hour                    / 23.0,
            self.day                     / 6.0,
            np.mean(self.demand_history) / 50.0,
            (self.competitor_price - 70) / 80.0,
            self.step_count              / self.max_steps,
        ], dtype=np.float32)


# ═══════════════════════════════════════════════════════════
#  VERIFY
# ═══════════════════════════════════════════════════════════
def verify_env_reward(seed=42):
    print("\n" + "="*58)
    print("  ENV SANITY CHECK (1 episode per multiplier)")
    print("="*58)
    print(f"  {'Mult':>6}  {'Price':>6}  {'AvgReward':>10}  {'AvgDemand':>10}  Bar")
    print("  " + "-"*55)

    results = []
    for mult in [0.8, 0.9, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.5]:
        env = RetailPricingEnv(seed=seed)
        env.reset()
        rewards = []; demands = []
        while True:
            _, r, done, info = env.step(mult, is_continuous=True)
            rewards.append(r); demands.append(info['demand'])
            if done: break
        results.append((mult, np.mean(rewards), np.mean(demands)))

    min_r = min(r for _,r,_ in results)
    max_r = max(r for _,r,_ in results)
    rng   = max(max_r - min_r, 1e-6)

    for mult, avg_r, avg_d in results:
        bar = "█" * max(0, int(30*(avg_r-min_r)/rng))
        flg = " ◄ PEAK" if avg_r == max_r else ""
        print(f"  x{mult:.2f}  ${100*mult:>5.0f}  {avg_r:>10.4f}  {avg_d:>10.2f}  {bar}{flg}")

    peak_mult = results[np.argmax([r for _,r,_ in results])][0]
    print(f"\n  Peak at x{peak_mult:.2f} (${100*peak_mult:.0f})")
    print("="*58)

    print("\n  1-step ground truth (hour=12, inv=50, comp=$95):")
    env2 = RetailPricingEnv(seed=seed)
    for mult in [0.9, 1.0, 1.1, 1.2, 1.3, 1.5]:
        env2.reset()
        env2.competitor_price = 95.0
        env2.hour = 12
        env2.inventory = 50
        _, r, _, info = env2.step(mult, is_continuous=True)
        print(f"    x{mult:.1f} (${100*mult:.0f}) | demand={info['demand']:5.1f} | "
              f"sold={info['sold']:3d} | reward={r:7.4f}")
    return True


# ═══════════════════════════════════════════════════════════
#  PLOTS
# ═══════════════════════════════════════════════════════════
def plot_training(history):
    fig,ax = plt.subplots(1,3,figsize=(15,4))
    fig.suptitle('DDPG Training', fontsize=13, fontweight='bold')

    def sm(a,w=20):
        return [np.mean(a[max(0,i-w):i+1]) for i in range(len(a))]

    eps = range(1, len(history['episode_rewards'])+1)

    ax[0].plot(eps, history['episode_rewards'], color='#1D9E7540', lw=1)
    ax[0].plot(eps, sm(history['episode_rewards']), color='#1D9E75', lw=2)
    ax[0].set_title('Episode Reward')
    ax[0].set_xlabel('Episode')
    ax[0].grid(alpha=0.3)

    ax[1].plot(eps, history['avg_prices'], color='#185FA5', lw=1.5)
    ax[1].axhline(y=100, color='#888', ls='--', lw=1, label='Base $100')
    ax[1].set_title('Avg Price Per Episode')
    ax[1].set_ylabel('Price ($)')
    ax[1].set_xlabel('Episode')
    ax[1].legend()
    ax[1].grid(alpha=0.3)

    if history['critic_losses']:
        cl = history['critic_losses']
        ax[2].plot(range(len(cl)), cl, color='#E24B4A40', lw=0.7)
        ax[2].plot(range(len(cl)), sm(cl,100), color='#E24B4A', lw=2)
        ax[2].set_title('Critic Loss')
        ax[2].set_xlabel('Update step')
        ax[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_price_behavior(env, agent):
    s=env.reset()
    prices,invs,dems,rews=[],[],[],[]

    agent.actor.eval()

    while True:
        a=agent.act(s,noise=False)
        s,r,done,info=env.step(a,is_continuous=True)

        prices.append(info['price'])
        invs.append(info['inventory'])
        dems.append(info['demand'])
        rews.append(r)

        if done:
            break

    agent.actor.train()

    fig,ax=plt.subplots(4,1,figsize=(14,10),sharex=True)
    fig.suptitle('DDPG Price Behavior (720 steps)', fontsize=13, fontweight='bold')

    st=range(len(prices))

    ax[0].plot(st,prices,color='#1D9E75',lw=1.5)
    ax[0].axhline(y=100,color='#888',ls='--',lw=1,label='Base $100')
    ax[0].set_ylabel('Price ($)')
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    ax[1].plot(st,dems,color='#9B59B6',lw=1.2,alpha=0.8)
    ax[1].set_ylabel('Demand')
    ax[1].grid(alpha=0.3)

    ax[2].fill_between(st,invs,alpha=0.3,color='#185FA5')
    ax[2].plot(st,invs,color='#185FA5',lw=1.3)
    ax[2].axhline(y=5,color='#E24B4A',ls='--',lw=1,label='Stockout')
    ax[2].set_ylabel('Inventory')
    ax[2].legend()
    ax[2].grid(alpha=0.3)

    cum=np.cumsum(rews)
    ax[3].plot(st,cum,color='#0F6E56',lw=2)
    ax[3].fill_between(st,cum,alpha=0.15,color='#0F6E56')
    ax[3].set_ylabel('Cumul. Reward')
    ax[3].set_xlabel('Step (hour)')
    ax[3].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    if len(set(dems)) > 1:
        corr = np.corrcoef(dems, prices)[0,1]
        print(f"\n  Price-Demand corr: {corr:+.3f}")
        print(f"  {'✅ Raises price at high demand' if corr>0.1 else '❌ No relationship'}")