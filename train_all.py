import numpy as np, random, json
from env_and_agents import RetailPricingEnv, QLearningAgent, FixedAgent, RuleAgent, DDPGAgent

np.random.seed(42); random.seed(42)

# ─── STEP 2: Q-LEARNING TRAINING ──────────────────────────────────────────────
print("="*50)
print("STEP 2: Training Q-Learning (300 episodes)...")
print("="*50)

env = RetailPricingEnv(seed=42)
agent = QLearningAgent(alpha=0.1, gamma=0.95, epsilon_start=1.0,
                       epsilon_end=0.05, epsilon_decay=0.995)
ql_log = []

for ep in range(1, 301):
    obs = env.reset()
    ep_actions, ep_prices, ep_demands, ep_inventories, ep_comp = [], [], [], [], []
    while True:
        action = agent.select_action(obs)
        nobs, reward, done, info = env.step(action)
        agent.update(obs, action, reward, nobs, done)
        obs = nobs
        ep_actions.append(action)
        ep_prices.append(info["price"])
        ep_demands.append(info["demand"])
        ep_inventories.append(info["inventory"])
        ep_comp.append(info["competitor_price"])
        if done: break
    agent.decay_epsilon()
    ql_log.append({
        "ep": ep, "profit": round(env.total_profit, 2),
        "epsilon": round(agent.epsilon, 4),
        "avg_price": round(float(np.mean(ep_prices)), 2),
        "avg_demand": round(float(np.mean(ep_demands)), 2),
        "avg_inv": round(float(np.mean(ep_inventories)), 2),
        "action_dist": np.bincount(ep_actions, minlength=5).tolist(),
    })
    if ep % 50 == 0:
        print(f"  Ep {ep:3d} | Profit ${env.total_profit:>9,.0f} | eps={agent.epsilon:.3f}")

print(f"\n  Best: ${max(r['profit'] for r in ql_log):,.0f}")
print(f"  Last-20 avg: ${np.mean([r['profit'] for r in ql_log[-20:]]):,.0f}")

final_actions = np.array(ql_log[-1]['action_dist'])
total_acts = np.zeros(5)
for r in ql_log: total_acts += np.array(r['action_dist'])

agent.save("ql_agent.npz")
with open("ql_training.json","w") as f:
    json.dump({"log": ql_log, "total_action_dist": total_acts.tolist()}, f)
print("  Saved -> ql_training.json, ql_agent.npz")

# ─── STEP 3: DDPG TRAINING (fast approximation) ───────────────────────────────
print("\n" + "="*50)
print("STEP 3: Training DDPG (200 episodes, fast mode)...")
print("="*50)

ddpg_env = RetailPricingEnv(seed=123)
ddpg = DDPGAgent(sdim=7, alo=0.7, ahi=1.5, gamma=0.99, tau=0.005, alr=2e-4, clr=2e-4, bs=64)
ddpg_log = []

# Warmup: fill buffer with random actions before training
print("  [Warmup] 500 random steps...")
obs = ddpg_env.reset()
for _ in range(500):
    a = random.uniform(0.7, 1.5)
    nobs, r, done, _ = ddpg_env.step(a, is_continuous=True)
    ddpg.store(obs, a, r, nobs, float(done))
    obs = nobs if not done else ddpg_env.reset()

for ep in range(1, 201):
    obs = ddpg_env.reset()
    ddpg.noise.reset()
    ep_prices, ep_demands = [], []
    t = 0
    while True:
        noise_on = ep > 10
        action = ddpg.act(obs, noise=noise_on)
        nobs, reward, done, info = ddpg_env.step(action, is_continuous=True)
        ddpg.store(obs, action, reward, nobs, float(done))
        obs = nobs
        ep_prices.append(info["price"])
        ep_demands.append(info["demand"])
        if ep > 15 and t % 5 == 0:
            ddpg.update()
        t += 1
        if done: break
    ddpg_log.append({
        "ep": ep, "profit": round(ddpg_env.total_profit, 2),
        "noise": round(ddpg.noise.sigma, 4),
        "avg_price": round(float(np.mean(ep_prices)), 2),
        "avg_demand": round(float(np.mean(ep_demands)), 2),
    })
    if ep % 40 == 0:
        print(f"  Ep {ep:3d} | Profit ${ddpg_env.total_profit:>9,.0f} | noise={ddpg.noise.sigma:.3f}")

print(f"\n  Best: ${max(r['profit'] for r in ddpg_log):,.0f}")
with open("ddpg_training.json","w") as f:
    json.dump({"log": ddpg_log}, f)
print("  Saved -> ddpg_training.json")

# ─── STEP 5: COMPARISON ───────────────────────────────────────────────────────
print("\n" + "="*50)
print("STEP 5: Profitability Comparison (100 eval episodes each)...")
print("="*50)

def eval_agent(ag, n=100, seed=999, discrete=True, greedy_ql=False):
    profits = []
    env2 = RetailPricingEnv(seed=seed)
    for _ in range(n):
        obs = env2.reset()
        while True:
            if greedy_ql: a = ag.select_action(obs, greedy=True)
            else: a = ag.select_action(obs)
            obs, _, done, _ = env2.step(a, is_continuous=not discrete)
            if done:
                profits.append(env2.total_profit)
                break
    return profits

fixed_profits = eval_agent(FixedAgent(), n=100)
rule_profits  = eval_agent(RuleAgent(),  n=100)
ql_profits    = eval_agent(agent, n=100, greedy_ql=True)

print(f"  Fixed    : ${np.mean(fixed_profits):>9,.0f} +/- ${np.std(fixed_profits):,.0f}")
print(f"  Rule     : ${np.mean(rule_profits):>9,.0f} +/- ${np.std(rule_profits):,.0f}")
print(f"  Q-Learning: ${np.mean(ql_profits):>9,.0f} +/- ${np.std(ql_profits):,.0f}")
print(f"\n  QL vs Fixed:    {(np.mean(ql_profits)-np.mean(fixed_profits))/abs(np.mean(fixed_profits))*100:+.1f}%")
print(f"  QL vs Rule:     {(np.mean(ql_profits)-np.mean(rule_profits))/abs(np.mean(rule_profits))*100:+.1f}%")

with open("comparison.json","w") as f:
    json.dump({
        "fixed":  {"mean": round(np.mean(fixed_profits),2), "std": round(np.std(fixed_profits),2), "all": [round(p,2) for p in fixed_profits]},
        "rule":   {"mean": round(np.mean(rule_profits),2),  "std": round(np.std(rule_profits),2),  "all": [round(p,2) for p in rule_profits]},
        "ql":     {"mean": round(np.mean(ql_profits),2),    "std": round(np.std(ql_profits),2),    "all": [round(p,2) for p in ql_profits]},
    }, f)
print("  Saved -> comparison.json")

# ─── STEP 6: SENSITIVITY ──────────────────────────────────────────────────────
print("\n" + "="*50)
print("STEP 6: Sensitivity Analysis...")
print("="*50)

configs = [
    {"label":"α=0.05", "alpha":0.05,"gamma":0.95,"decay":0.995},
    {"label":"α=0.10 (base)","alpha":0.10,"gamma":0.95,"decay":0.995},
    {"label":"α=0.20", "alpha":0.20,"gamma":0.95,"decay":0.995},
    {"label":"γ=0.80", "alpha":0.10,"gamma":0.80,"decay":0.995},
    {"label":"γ=0.99", "alpha":0.10,"gamma":0.99,"decay":0.995},
    {"label":"ε-decay=0.990","alpha":0.10,"gamma":0.95,"decay":0.990},
    {"label":"ε-decay=0.999","alpha":0.10,"gamma":0.95,"decay":0.999},
]
sens_results = []
for cfg in configs:
    senv = RetailPricingEnv(seed=7)
    sag  = QLearningAgent(alpha=cfg["alpha"], gamma=cfg["gamma"],
                          epsilon_decay=cfg["decay"], epsilon_end=0.05)
    curve = []
    for ep in range(200):
        obs = senv.reset()
        while True:
            a = sag.select_action(obs)
            nobs,r,done,_ = senv.step(a)
            sag.update(obs,a,r,nobs,done)
            obs=nobs
            if done: break
        sag.decay_epsilon()
        curve.append(round(senv.total_profit,2))
    last30 = curve[-30:]
    entry = {**cfg, "mean": round(float(np.mean(last30)),2),
             "std": round(float(np.std(last30)),2), "curve": curve}
    sens_results.append(entry)
    print(f"  {cfg['label']:<22} -> ${entry['mean']:>9,.0f} +/- ${entry['std']:,.0f}")

with open("sensitivity.json","w") as f:
    json.dump(sens_results, f)
print("  Saved -> sensitivity.json")

# ─── STEP 6b: DDPG SENSITIVITY ────────────────────────────────────────────────
print("\n" + "="*50)
print("STEP 6b: DDPG Hyperparameter Sensitivity (50 episodes each)...")
print("="*50)

ddpg_sens_configs = [
    {"label": "alr=1e-4,clr=3e-4 (base)", "alr": 1e-4, "clr": 3e-4, "tau": 0.005, "bs": 128},
    {"label": "alr=3e-4,clr=3e-4",        "alr": 3e-4, "clr": 3e-4, "tau": 0.005, "bs": 128},
    {"label": "alr=1e-4,clr=1e-3",        "alr": 1e-4, "clr": 1e-3, "tau": 0.005, "bs": 128},
    {"label": "tau=0.001",                 "alr": 1e-4, "clr": 3e-4, "tau": 0.001, "bs": 128},
    {"label": "tau=0.01",                  "alr": 1e-4, "clr": 3e-4, "tau": 0.010, "bs": 128},
    {"label": "bs=64",                     "alr": 1e-4, "clr": 3e-4, "tau": 0.005, "bs":  64},
    {"label": "bs=256",                    "alr": 1e-4, "clr": 3e-4, "tau": 0.005, "bs": 256},
]
ddpg_sens_results = []
for cfg in ddpg_sens_configs:
    senv = RetailPricingEnv(seed=7)
    sag  = DDPGAgent(sdim=7, alr=cfg["alr"], clr=cfg["clr"], tau=cfg["tau"], bs=cfg["bs"])
    curve = []
    # short warmup
    obs = senv.reset()
    for _ in range(300):
        a = random.uniform(0.7, 1.5)
        nobs, r, done, _ = senv.step(a, is_continuous=True)
        sag.store(obs, a, r, nobs, float(done))
        obs = nobs if not done else senv.reset()
    for ep in range(50):
        obs = senv.reset()
        sag.noise.reset()
        while True:
            a = sag.act(obs, noise=True)
            nobs, r, done, _ = senv.step(a, is_continuous=True)
            sag.store(obs, a, r, nobs, float(done))
            sag.update()
            obs = nobs
            if done: break
        curve.append(round(senv.total_profit, 2))
    last15 = curve[-15:]
    entry = {**cfg, "mean": round(float(np.mean(last15)), 2),
             "std": round(float(np.std(last15)), 2), "curve": curve}
    ddpg_sens_results.append(entry)
    print(f"  {cfg['label']:<30} -> ${entry['mean']:>9,.0f} +/- ${entry['std']:,.0f}")

with open("ddpg_sensitivity.json", "w") as f:
    json.dump(ddpg_sens_results, f)
print("  Saved -> ddpg_sensitivity.json")

# ─── STEP 5b: ADD DDPG TO COMPARISON ─────────────────────────────────────────
print("\n" + "="*50)
print("STEP 5b: Evaluating trained DDPG agent (100 episodes)...")
print("="*50)
import os
if os.path.exists("ddpg_best.pth"):
    ddpg_eval = DDPGAgent.load("ddpg_best.pth")
    ddpg_profits = []
    eval_env = RetailPricingEnv(seed=999)
    for _ in range(100):
        obs = eval_env.reset()
        while True:
            a = ddpg_eval.act(obs, noise=False)
            obs, _, done, _ = eval_env.step(a, is_continuous=True)
            if done:
                ddpg_profits.append(eval_env.total_profit)
                break
    print(f"  DDPG: ${np.mean(ddpg_profits):>9,.0f} +/- ${np.std(ddpg_profits):,.0f}")
    # update comparison.json with ddpg entry
    with open("comparison.json") as f:
        comp = json.load(f)
    comp["ddpg"] = {
        "mean": round(float(np.mean(ddpg_profits)), 2),
        "std":  round(float(np.std(ddpg_profits)), 2),
        "all":  [round(p, 2) for p in ddpg_profits],
    }
    with open("comparison.json", "w") as f:
        json.dump(comp, f)
    print("  Updated comparison.json with DDPG results")
else:
    print("  ddpg_best.pth not found — skipping DDPG eval")

import subprocess, sys
for script in ("build_dash_data.py", "build_dashboard.py", "generate_report.py"):
    print(f"\nRunning {script}...")
    subprocess.run([sys.executable, script], check=True)
print("\nDone. Charts: rl_dashboard.html")
print("Live Simulator + Competitors: python dashboard_server.py -> http://127.0.0.1:5050")
