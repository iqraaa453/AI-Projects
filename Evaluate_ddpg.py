# ═══════════════════════════════════════════════════════════
#  TEST DDPG MODEL (VS CODE VERSION)
# ═══════════════════════════════════════════════════════════

import numpy as np
import random
import torch
import matplotlib.pyplot as plt

# ✅ Import your environment + agent
from Simulation import RetailPricingEnv
from DDPG import DDPGAgent


# ==========================================================
# DEVICE
# ==========================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\nUsing device: {DEVICE}")


# ==========================================================
# LOAD MODEL
# ==========================================================

env = RetailPricingEnv(seed=42)

agent = DDPGAgent(sd=env.state_dim)

checkpoint = torch.load(
    "ddpg_best.pth",
    map_location=DEVICE
)

agent.actor.load_state_dict(checkpoint['actor'])
agent.critic.load_state_dict(checkpoint['critic'])

agent.actor.eval()
agent.critic.eval()

print("\n✅ Model loaded successfully!")


# ==========================================================
# TESTING
# ==========================================================

state = env.reset()

prices   = []
demands  = []
inventory = []
rewards  = []

total_reward = 0

while True:

    # No exploration noise during testing
    action = agent.act(state, noise=False)

    next_state, reward, done, info = env.step(
        action,
        is_continuous=True
    )

    prices.append(info['price'])
    demands.append(info['demand'])
    inventory.append(info['inventory'])
    rewards.append(reward)

    total_reward += reward

    state = next_state

    if done:
        break


# ==========================================================
# RESULTS
# ==========================================================

print("\n" + "="*55)
print("             TEST RESULTS")
print("="*55)

print(f"Total Reward     : {total_reward:.2f}")
print(f"Average Price    : ${np.mean(prices):.2f}")
print(f"Average Demand   : {np.mean(demands):.2f}")
print(f"Final Inventory  : {inventory[-1]}")

print("="*55)


# ==========================================================
# PLOTS
# ==========================================================

steps = range(len(prices))

fig, ax = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

fig.suptitle(
    "DDPG Testing Results",
    fontsize=14,
    fontweight='bold'
)

# Price
ax[0].plot(steps, prices, lw=1.5)
ax[0].axhline(y=100, linestyle='--')
ax[0].set_ylabel("Price")
ax[0].grid(alpha=0.3)

# Demand
ax[1].plot(steps, demands, lw=1.2)
ax[1].set_ylabel("Demand")
ax[1].grid(alpha=0.3)

# Inventory
ax[2].plot(steps, inventory, lw=1.2)
ax[2].axhline(y=5, linestyle='--')
ax[2].set_ylabel("Inventory")
ax[2].grid(alpha=0.3)

# Cumulative Reward
cum_reward = np.cumsum(rewards)

ax[3].plot(steps, cum_reward, lw=2)
ax[3].set_ylabel("Cum Reward")
ax[3].set_xlabel("Steps")
ax[3].grid(alpha=0.3)

plt.tight_layout()
plt.show()


# ==========================================================
# OPTIONAL ANALYSIS
# ==========================================================

corr = np.corrcoef(prices, demands)[0,1]

print(f"\nPrice-Demand Correlation: {corr:+.3f}")

if corr > 0:
    print("Agent raises prices during high demand")
else:
    print("Weak pricing strategy relationship")