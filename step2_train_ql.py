"""
STEP 2 — Q-Learning agent (discrete pricing)
Run: python step2_train_ql.py
"""
import json
import numpy as np
import random
from env_and_agents import RetailPricingEnv, QLearningAgent

np.random.seed(42)
random.seed(42)

EPISODES = 300
ALPHA = 0.1
GAMMA = 0.95
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995

print("=" * 50)
print("STEP 2: Training Q-Learning")
print(f"  Episodes: {EPISODES} | alpha={ALPHA} | gamma={GAMMA}")
print("=" * 50)

env = RetailPricingEnv(seed=42)
agent = QLearningAgent(
    alpha=ALPHA,
    gamma=GAMMA,
    epsilon_start=EPS_START,
    epsilon_end=EPS_END,
    epsilon_decay=EPS_DECAY,
)
log = []

for ep in range(1, EPISODES + 1):
    obs = env.reset()
    actions, prices = [], []
    while True:
        action = agent.select_action(obs)
        nobs, reward, done, info = env.step(action)
        agent.update(obs, action, reward, nobs, done)
        obs = nobs
        actions.append(action)
        prices.append(info["price"])
        if done:
            break
    agent.decay_epsilon()
    log.append({
        "ep": ep,
        "profit": round(env.total_profit, 2),
        "epsilon": round(agent.epsilon, 4),
        "avg_price": round(float(np.mean(prices)), 2),
        "action_dist": np.bincount(actions, minlength=5).tolist(),
    })
    if ep % 50 == 0:
        print(f"  Ep {ep:3d} | Profit ${env.total_profit:>9,.0f} | eps={agent.epsilon:.3f}")

total_acts = np.zeros(5)
for r in log:
    total_acts += np.array(r["action_dist"])

agent.save("ql_agent.npz")
with open("ql_training.json", "w") as f:
    json.dump({"log": log, "total_action_dist": total_acts.tolist()}, f)

print(f"\n  Best profit : ${max(r['profit'] for r in log):,.0f}")
print(f"  Last-20 avg : ${np.mean([r['profit'] for r in log[-20:]]):,.0f}")
print("  Saved -> ql_agent.npz, ql_training.json")
print("\nNext: python step5_compare.py")
