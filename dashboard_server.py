"""
Live dashboard backend — real RetailPricingEnv + trained agents.
Run: python dashboard_server.py
Open: http://127.0.0.1:5050
"""
import os
import random
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from env_and_agents import (
    RetailPricingEnv,
    QLearningAgent,
    FixedAgent,
    RuleAgent,
    DDPGAgent,
)

ROOT = Path(__file__).parent
QL_PATH   = ROOT / "ql_agent.npz"
DDPG_PATH = ROOT / "ddpg_best.pth"

app = Flask(__name__, static_folder=str(ROOT), static_url_path="")


def _agents():
    agents = {
        "fixed": FixedAgent(),
        "rule":  RuleAgent(),
    }
    if QL_PATH.exists():
        agents["ql"] = QLearningAgent.load(str(QL_PATH))
    if DDPG_PATH.exists():
        agents["ddpg"] = DDPGAgent.load(str(DDPG_PATH))
    return agents


AGENTS = _agents()

# Live market session (competitor tab) — steps real env with Q-Learning
market = {
    "env": RetailPricingEnv(),
    "history": [],
    "peer_history": {},
    "max_hist": 30,
}

# Interactive simulator session
sim = {"env": None, "strategy": "rule", "trace": []}


def _pick_action(strategy, obs, agents):
    if strategy == "fixed":
        return agents["fixed"].select_action(obs), False
    if strategy == "rule":
        return agents["rule"].select_action(obs), False
    if strategy == "ql":
        if "ql" not in agents:
            raise ValueError("ql_agent.npz not found — run step2_train_ql.py")
        return agents["ql"].select_action(obs, greedy=True), False
    if strategy == "ddpg":
        if "ddpg" not in agents:
            raise ValueError("ddpg_best.pth not found — run train_all.py")
        return agents["ddpg"].act(obs, noise=False), True   # continuous
    if strategy == "random":
        return random.randint(0, 4), False
    raise ValueError(f"Unknown strategy: {strategy}")


def _state_dict(env, info, reward=0.0):
    return {
        "inventory": env.inventory,
        "our_price": round(env.current_price, 2),
        "competitor_price": round(env.competitor_price, 2),
        "demand": info.get("demand", 0),
        "sold": info.get("sold", 0),
        "profit": round(env.total_profit, 2),
        "reward": round(float(reward), 2),
        "hour": env.hour,
        "day": env.day,
        "step": env.step_count,
        "age_pct": info.get("product_age_pct", 0),
    }


def _peer_competitors(primary: float, step: int):
    """Peer shops anchored to env competitor price (not independent random JS)."""
    names = ["RivalMart", "PriceKing", "QuickShop", "BudgetHub", "LuxStore"]
    factors = [0.94, 0.88, 1.02, 0.78, 1.15]
    peers = []
    for name, f in zip(names, factors):
        wobble = np.sin(step * 0.17 + hash(name) % 7) * 1.5
        price = float(np.clip(primary * f + wobble, 65, 140))
        peers.append({"name": name, "price": round(price, 2)})
    return peers


@app.route("/")
def index():
    return send_from_directory(ROOT, "rl_dashboard.html")


@app.route("/api/health")
def health():
    return jsonify({
        "ok": True,
        "ql_loaded": QL_PATH.exists(),
        "strategies": list(AGENTS.keys()) + ["random", "ddpg"],
    })


@app.route("/api/market/reset", methods=["POST"])
def market_reset():
    market["env"] = RetailPricingEnv()
    market["history"] = []
    market["peer_history"] = {}
    return jsonify({"ok": True})


@app.route("/api/market/tick")
def market_tick():
    """Advance live market one hour using trained Q-Learning (or rule if no QL)."""
    env = market["env"]
    agents = AGENTS
    strat = "ddpg" if "ddpg" in agents else ("ql" if "ql" in agents else "rule")

    if env.step_count >= env.max_steps:
        env.reset()

    obs = env._get_observation()
    action, cont = _pick_action(strat, obs, agents)
    nobs, reward, done, info = env.step(action, is_continuous=cont)
    if done:
        obs = env.reset()
        action, cont = _pick_action(strat, obs, agents)
        nobs, reward, done, info = env.step(action, is_continuous=cont)

    tick = _state_dict(env, info, reward)
    tick["strategy"] = strat
    market["history"].append(tick)
    if len(market["history"]) > market["max_hist"]:
        market["history"] = market["history"][-market["max_hist"] :]

    peers = _peer_competitors(env.competitor_price, env.step_count)
    for p in peers:
        market["peer_history"].setdefault(p["name"], []).append(p["price"])
        if len(market["peer_history"][p["name"]]) > market["max_hist"]:
            market["peer_history"][p["name"]] = market["peer_history"][p["name"]][-market["max_hist"] :]

    prices = [p["price"] for p in peers]
    return jsonify({
        "tick": tick,
        "peers": peers,
        "our_price": tick["our_price"],
        "market_avg": round(float(np.mean(prices + [tick["our_price"]])), 2),
        "market_min": round(float(min(prices + [tick["our_price"]])), 2),
        "history": market["history"],
        "peer_history": market["peer_history"],
    })


@app.route("/api/sim/reset", methods=["POST"])
def sim_reset():
    data = request.get_json(silent=True) or {}
    sim["strategy"] = data.get("strategy", "rule")
    sim["env"] = RetailPricingEnv()
    sim["env"].reset()
    sim["trace"] = []
    return jsonify({"ok": True, "strategy": sim["strategy"]})


@app.route("/api/sim/run", methods=["POST"])
def sim_run():
    data = request.get_json(silent=True) or {}
    strategy = data.get("strategy", "rule")
    steps = int(data.get("steps", 120))
    steps = max(1, min(steps, 720))

    env = RetailPricingEnv()
    env.reset()
    agents = AGENTS
    log = []
    prices = []
    demands = []
    rewards = []

    obs = env.reset()
    for t in range(steps):
        action, cont = _pick_action(strategy, obs, agents)
        nobs, reward, done, info = env.step(action, is_continuous=cont)
        obs = nobs
        prices.append(info["price"])
        demands.append(info["demand"])
        rewards.append(float(reward))
        if t % 24 == 0:
            log.append(
                f"Day {t // 24 + 1} | ${info['price']} | sold {info['sold']} | "
                f"inv {info['inventory']} | profit ${env.total_profit:,.0f}"
            )
        if done:
            break

    return jsonify({
        "strategy": strategy,
        "steps_run": len(prices),
        "total_profit": round(env.total_profit, 2),
        "avg_price": round(float(np.mean(prices)), 2) if prices else 0,
        "log": log[-12:],
        "prices": prices,
        "demands": demands,
        "rewards": rewards,
        "final": _state_dict(env, info if prices else {}, 0),
    })


@app.route("/api/sim/step", methods=["POST"])
def sim_step():
    """Single-step live simulation (optional streaming)."""
    data = request.get_json(silent=True) or {}
    if sim["env"] is None:
        sim_reset()
    if data.get("strategy"):
        sim["strategy"] = data["strategy"]

    env = sim["env"]
    agents = AGENTS
    obs = env._get_observation() if env.step_count else env.reset()
    if env.step_count == 0:
        obs = env.reset()

    action, cont = _pick_action(sim["strategy"], obs, agents)
    nobs, reward, done, info = env.step(action, is_continuous=cont)
    tick = _state_dict(env, info, reward)
    sim["trace"].append(tick)
    if done:
        env.reset()

    return jsonify({"tick": tick, "done": done, "trace": sim["trace"][-60:]})


def main():
    port = int(os.environ.get("PORT", 5050))
    if not QL_PATH.exists():
        print("WARNING: ql_agent.npz missing — run: python step2_train_ql.py")
    print(f"Live dashboard: http://127.0.0.1:{port}")
    print("Uses real RetailPricingEnv + trained agents for Simulator & Competitor tabs.")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
