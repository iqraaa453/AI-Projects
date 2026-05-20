"""
STEP 5 — Fixed vs Rule-Based vs Q-Learning vs DDPG comparison
Requires: ql_agent.npz  (from step2_train_ql.py / train_all.py)
          ddpg_best.pth (from train_all.py)
Run: python step5_compare.py
"""
import json
import os
import numpy as np
from env_and_agents import RetailPricingEnv, QLearningAgent, FixedAgent, RuleAgent, DDPGAgent

EVAL_EPISODES = 100
SEED = 999


def eval_discrete(agent, n=EVAL_EPISODES, greedy_ql=False):
    profits = []
    env = RetailPricingEnv(seed=SEED)
    for _ in range(n):
        obs = env.reset()
        while True:
            a = agent.select_action(obs, greedy=True) if greedy_ql else agent.select_action(obs)
            obs, _, done, _ = env.step(a)
            if done:
                profits.append(env.total_profit)
                break
    return profits


def eval_ddpg(agent, n=EVAL_EPISODES):
    profits = []
    env = RetailPricingEnv(seed=SEED)
    for _ in range(n):
        obs = env.reset()
        while True:
            a = agent.act(obs, noise=False)
            obs, _, done, _ = env.step(a, is_continuous=True)
            if done:
                profits.append(env.total_profit)
                break
    return profits


def main():
    print("=" * 55)
    print("STEP 5: Profitability Comparison")
    print(f"  {EVAL_EPISODES} evaluation episodes per strategy")
    print("=" * 55)

    if not os.path.exists("ql_agent.npz"):
        print("ERROR: ql_agent.npz not found. Run step2_train_ql.py first.")
        return

    ql_agent = QLearningAgent.load("ql_agent.npz")
    fixed_p  = eval_discrete(FixedAgent())
    rule_p   = eval_discrete(RuleAgent())
    ql_p     = eval_discrete(ql_agent, greedy_ql=True)

    ddpg_p = None
    if os.path.exists("ddpg_best.pth"):
        ddpg_agent = DDPGAgent.load("ddpg_best.pth")
        ddpg_p = eval_ddpg(ddpg_agent)
    else:
        print("  WARNING: ddpg_best.pth not found — DDPG skipped.")

    rows = [
        ("Fixed Pricing", fixed_p),
        ("Rule-Based",    rule_p),
        ("Q-Learning",    ql_p),
    ]
    if ddpg_p is not None:
        rows.append(("DDPG", ddpg_p))

    print()
    for name, profits in rows:
        print(f"  {name:<14}: ${np.mean(profits):>9,.0f}  ±${np.std(profits):,.0f}")

    fm = np.mean(fixed_p)
    qm = np.mean(ql_p)
    rm = np.mean(rule_p)
    print(f"\n  QL   vs Fixed:  {(qm - fm) / abs(fm) * 100:+.1f}%")
    print(f"  QL   vs Rule:   {(qm - rm) / abs(rm) * 100:+.1f}%")
    if ddpg_p is not None:
        dm = np.mean(ddpg_p)
        print(f"  DDPG vs Fixed:  {(dm - fm) / abs(fm) * 100:+.1f}%")
        print(f"  DDPG vs Rule:   {(dm - rm) / abs(rm) * 100:+.1f}%")
        print(f"  DDPG vs QL:     {(dm - qm) / abs(qm) * 100:+.1f}%")

    out = {
        "fixed": {"mean": round(float(np.mean(fixed_p)), 2), "std": round(float(np.std(fixed_p)), 2),
                  "all":  [round(p, 2) for p in fixed_p]},
        "rule":  {"mean": round(float(np.mean(rule_p)), 2),  "std": round(float(np.std(rule_p)), 2),
                  "all":  [round(p, 2) for p in rule_p]},
        "ql":    {"mean": round(float(np.mean(ql_p)), 2),    "std": round(float(np.std(ql_p)), 2),
                  "all":  [round(p, 2) for p in ql_p]},
    }
    if ddpg_p is not None:
        out["ddpg"] = {"mean": round(float(np.mean(ddpg_p)), 2), "std": round(float(np.std(ddpg_p)), 2),
                       "all":  [round(p, 2) for p in ddpg_p]}

    with open("comparison.json", "w") as f:
        json.dump(out, f)
    print("\n  Saved -> comparison.json")
    print("  Next: python build_dash_data.py && python build_dashboard.py")


if __name__ == "__main__":
    main()
