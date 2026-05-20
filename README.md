# Dynamic Pricing Optimization (RL)

Retail pricing simulation with **Q-Learning** (discrete prices) and **DDPG** (continuous prices, PyTorch), plus a live dashboard, competitor tracker, profitability comparison, and sensitivity analysis.

## Project structure

| File | Purpose |
|------|---------|
| `env_and_agents.py` | `RetailPricingEnv`, Q-Learning, DDPG (PyTorch), Fixed/Rule agents |
| `RL_Env_Simulation.ipynb` | Environment demo & sanity tests |
| `step2_train_ql.py` | Train Q-Learning agent |
| `step5_compare.py` | Fixed vs Rule vs QL vs DDPG profitability comparison |
| `train_all.py` | Full pipeline (steps 2–6 + DDPG sensitivity) |
| `build_dash_data.py` | Merge JSON → `dash_data.json` |
| `build_dashboard.py` | Build `rl_dashboard.html` |
| `dashboard_server.py` | Live simulator + competitor tracker (Flask) |

## Quick start (Windows)

```powershell
cd "file-location"
pip install -r requirements.txt
run_project.bat
```

## Assignment mapping

| Step | Task | Script |
|------|------|--------|
| 1 | Simulation environment | `RL_Env_Simulation.ipynb` / `env_and_agents.py` |
| 2 | Q-Learning training | `step2_train_ql.py` |
| 3 | DDPG training (PyTorch) | `train_all.py` (Step 3 section) |
| 4 | Dashboard | `build_dashboard.py` → `rl_dashboard.html` |
| 5 | Fixed vs Rule vs QL vs DDPG | `step5_compare.py` |

## Environment details

- **Discrete actions:** 0.8, 0.9, 1.0, 1.1, 1.2 × base price ($100)
- **Continuous (DDPG):** multiplier ∈ [0.7, 1.5] via trained Actor network
- **State:** inventory, price, competitor price, hour, day, 7-day demand avg, product age %
- **Reward:** (revenue − COGS) − holding cost − stockout penalty − volatility penalty
- **Demand:** price-elastic vs competitor; peak-hour boost; weekend boost

## DDPG architecture

| Component | Architecture |
|-----------|-------------|
| **Actor** | 7 → 256 (LayerNorm+ReLU) → 256 (LayerNorm+ReLU) → 128 (ReLU) → 1 (Sigmoid→[0.7,1.5]) |
| **Critic** | State branch: 7→256 (LN+ReLU); concat action → 256→128→1 |
| **Noise** | Ornstein–Uhlenbeck with σ-decay |
| **Optimiser** | Adam; grad clipping 0.5 (actor), 1.0 (critic); soft update τ=0.005 |

## Outputs

After `python train_all.py`:

| File | Contents |
|------|----------|
| `ql_agent.npz` | Saved Q-table |
| `ddpg_best.pth` | Best PyTorch DDPG weights |
| `ql_training.json` | Q-Learning learning curve |
| `ddpg_training.json` | DDPG learning curve |
| `comparison.json` | 100-episode eval: Fixed / Rule / QL / DDPG |
| `sensitivity.json` | 7 Q-Learning hyperparameter configs |
| `ddpg_sensitivity.json` | 7 DDPG hyperparameter configs |
| `rl_dashboard.html` | Interactive static dashboard |

## Live dashboard strategies

The live simulator (`python dashboard_server.py`) supports:
- `fixed` — always $100
- `rule` — heuristic rule agent
- `ql` — trained Q-Learning agent (requires `ql_agent.npz`)
- `ddpg` — trained DDPG agent (requires `ddpg_best.pth`)
- `random` — random discrete action
