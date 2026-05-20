import numpy as np, random, json
from env_and_agents import RetailPricingEnv, DDPGAgent

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
    obs = senv.reset()
    for _ in range(300):
        a = random.uniform(0.7, 1.5)
        nobs, r, done, _ = senv.step(a, is_continuous=True)
        sag.store(obs, a, r, nobs, float(done))
        obs = nobs if not done else senv.reset()
    t = 0
    for ep in range(10):   # reduced from 50
        obs = senv.reset()
        sag.noise.reset()
        while True:
            a = sag.act(obs, noise=True)
            nobs, r, done, _ = senv.step(a, is_continuous=True)
            sag.store(obs, a, r, nobs, float(done))
            if t % 5 == 0:   # update every 5 steps
                sag.update()
            t += 1
            obs = nobs
            if done: break
        curve.append(round(senv.total_profit, 2))
    last15 = curve[-10:]
    entry = {**cfg, "mean": round(float(np.mean(last15)), 2),
             "std": round(float(np.std(last15)), 2), "curve": curve}
    ddpg_sens_results.append(entry)
    print(f"  {cfg['label']:<30} -> ${entry['mean']:>9,.0f} +/- ${entry['std']:,.0f}")

with open("ddpg_sensitivity.json", "w") as f:
    json.dump(ddpg_sens_results, f)
print("Saved -> ddpg_sensitivity.json")