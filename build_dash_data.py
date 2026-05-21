import json
import os

def moving_avg(data, window=10):
    result = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        result.append(round(sum(data[start:i+1]) / (i - start + 1), 2))
    return result

def main():
    # 1. Load Q-Learning training log
    with open("ql_training.json") as f:
        ql_raw = json.load(f)
    ql_log = ql_raw["log"]
    total_action_dist = ql_raw.get("total_action_dist", [0]*5)

    # 2. Load Comparison data
    with open("comparison.json") as f:
        comparison = json.load(f)

    # 3. Load QL Sensitivity
    with open("sensitivity.json") as f:
        ql_sensitivity = json.load(f)

    # 4. Safely load DDPG training if it exists
    ddpg_log = []
    if os.path.exists("ddpg_training.json"):
        with open("ddpg_training.json") as f:
            ddpg_log = json.load(f)["log"]

    # 5. Safely load DDPG sensitivity if it exists
    ddpg_sensitivity = []
    if os.path.exists("ddpg_sensitivity.json"):
        with open("ddpg_sensitivity.json") as f:
            ddpg_sensitivity = json.load(f)

    # --- Derive all flattened keys expected by the dashboard JS ---

    ql_profits  = [ep["profit"]    for ep in ql_log]
    ql_epsilons = [ep["epsilon"]   for ep in ql_log]
    ql_prices   = [ep["avg_price"] for ep in ql_log]
    ql_smooth   = moving_avg(ql_profits, 10)

    ddpg_profits = [ep["profit"]              for ep in ddpg_log]
    ddpg_noise   = [ep.get("noise", 0)        for ep in ddpg_log]
    ddpg_prices  = [ep.get("avg_price", 0)    for ep in ddpg_log]
    ddpg_smooth  = moving_avg(ddpg_profits, 10) if ddpg_profits else []

    # Flatten comparison
    comp = comparison
    comp_fixed_mean = comp["fixed"]["mean"]
    comp_fixed_std  = comp["fixed"]["std"]
    comp_fixed_all  = comp["fixed"]["all"]
    comp_rule_mean  = comp["rule"]["mean"]
    comp_rule_std   = comp["rule"]["std"]
    comp_rule_all   = comp["rule"]["all"]
    comp_ql_mean    = comp["ql"]["mean"]
    comp_ql_std     = comp["ql"]["std"]
    comp_ql_all     = comp["ql"]["all"]
    comp_ddpg_mean  = comp.get("ddpg", {}).get("mean", 0)
    comp_ddpg_std   = comp.get("ddpg", {}).get("std", 0)
    comp_ddpg_all   = comp.get("ddpg", {}).get("all", [])

    # Flatten QL sensitivity
    sens_labels = [s["label"] for s in ql_sensitivity]
    sens_means  = [s["mean"]  for s in ql_sensitivity]
    sens_stds   = [s["std"]   for s in ql_sensitivity]
    sens_curves = [s["curve"] for s in ql_sensitivity]
    sens_meta   = [{"alpha": s["alpha"], "gamma": s["gamma"], "decay": s["decay"]} for s in ql_sensitivity]

    # Flatten DDPG sensitivity
    ddpg_sens_labels = [s["label"] for s in ddpg_sensitivity] if ddpg_sensitivity else []
    ddpg_sens_means  = [s["mean"]  for s in ddpg_sensitivity] if ddpg_sensitivity else []

    # 6. Structure the full dashboard data with all flat keys
    dash_data = {
        # Raw arrays (kept for backwards compat)
        "ql_profits":      ql_profits,
        "ql_epsilons":     ql_epsilons,
        "ql_avg_prices":   ql_prices,
        "ddpg_profits":    ddpg_profits,
        "ddpg_noise":      ddpg_noise,
        "ddpg_avg_prices": ddpg_prices,
        "comparison":      comparison,
        "sensitivity":     ql_sensitivity,
        "ddpg_sensitivity": ddpg_sensitivity,

        # Derived / flattened keys expected by the JS
        "ql_smooth":       ql_smooth,
        "ql_eps":          ql_epsilons,
        "ql_prices":       ql_prices,
        "ddpg_smooth":     ddpg_smooth,
        "ddpg_prices":     ddpg_prices,
        "action_dist":     total_action_dist,

        "comp_fixed_mean": comp_fixed_mean,
        "comp_fixed_std":  comp_fixed_std,
        "comp_fixed_all":  comp_fixed_all,
        "comp_rule_mean":  comp_rule_mean,
        "comp_rule_std":   comp_rule_std,
        "comp_rule_all":   comp_rule_all,
        "comp_ql_mean":    comp_ql_mean,
        "comp_ql_std":     comp_ql_std,
        "comp_ql_all":     comp_ql_all,
        "comp_ddpg_mean":  comp_ddpg_mean,
        "comp_ddpg_std":   comp_ddpg_std,
        "comp_ddpg_all":   comp_ddpg_all,

        "sens_labels": sens_labels,
        "sens_means":  sens_means,
        "sens_stds":   sens_stds,
        "sens_curves": sens_curves,
        "sens_meta":   sens_meta,

        "ddpg_sens_labels": ddpg_sens_labels,
        "ddpg_sens_means":  ddpg_sens_means,
    }

    # 7. Write out to dash_data.json
    with open("dash_data.json", "w") as f:
        json.dump(dash_data, f)

    print("Successfully built -> dash_data.json")

if __name__ == "__main__":
    main()
