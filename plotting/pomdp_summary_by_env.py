"""Plots a summary of MMER ordered by environment"""

import plotlib as pl
import matplotlib.pyplot as plt
import seaborn as sb
import os
from matplotlib import ticker 
import getpass


WORKDIR = f"/Users/{getpass.getuser()}/data/laplace"
SAVEDIR = WORKDIR + "/plots"
AGG_MAP = {
    "Base Env": "first", 
    "Difficulty": "first", 
    "Model": "first",
    "MMER": "mean",
    "Normalized MMER": "mean",
}
MAX_MAP = {
    "Base Env": "first", 
    "Difficulty": "first", 
    "Model": "first",
    "MMER": "max",
    "Normalized MMER": "max",
}
os.makedirs(SAVEDIR, exist_ok=True)

def nameit(row):
    if 'sac' in row["name"]:
        alg = 'SAC'
    else:
        alg = 'TD3'
    return row["Model"] + "+" + alg
    

def main():
    env_names = {
        "AntBLT-V": "Ant-V",
        "WalkerBLT-V": "Walker-V",
        "HalfCheetahBLT-V": "HalfCheetah-V",
        "HopperBLT-V": "Hopper-V",
        "AntBLT-P": "Ant-P",
        "WalkerBLT-P": "Walker-P",
        "HalfCheetahBLT-P": "HalfCheetah-P",
        "HopperBLT-P": "Hopper-P",
    }
    projects = {"FFM_pomdp": [1_500_000, 0], "FFM_extra_pomdp": [1_500_000, 0]}
    runs, summary = pl.build_projects(projects, WORKDIR, process_fn=pl.process_run_pomdp, multiprocess=True, clean=False,
    x_key="z/rl_steps",
    metric_keys={
        "metrics/return_eval_total": "Episodic Reward",
        "z/rl_steps": "Env Steps",
    },
    recategorize_keys=["Env", "Model"],
    )
    summary = summary.replace(env_names)
    summary = summary.replace({"GRU/LSTM": "RNN"})

    # Plotting
    sb.set()
    sb.set_style("darkgrid")
    sb.set_context('paper')

    fig0, ax0 = plt.subplots(figsize=(10, 2))
    # TODO: use pointplot
    #order = summary.groupby('Env')['Reward'].mean()
    order = summary.groupby("Env").mean(numeric_only=True).sort_values("Reward").index
    summary["Model+Alg"] = summary.apply(nameit, axis=1)
    #summary[(summary["Model"] == "GRU/LSTM") & (summary["name"].str.contains("sac")), "Model+Alg."] = "GRU/LSTM+SAC"
    #summary[(summary["Model"] == "GRU/LSTM") & (summary["name"].str.contains("td3")), "Model+Alg."] = "GRU/LSTM+TD3"
    #summary[(summary["Model"] == "FFM") & (summary["name"].str.contains("sac")), "Model+Alg."] = "FFM+SAC"
    #summary[(summary["Model"] == "FFM") & (summary["name"].str.contains("td3")), "Model+Alg."] = "FFM+TD3"
    sb.barplot(data=summary, x="Env", y="Reward", hue="Model+Alg", ax=ax0, order=order, palette="Paired", hue_order=[
        "RNN+SAC",
        "FFM+SAC",
        "RNN+TD3",
        "FFM+TD3"
        ])
    plt.tight_layout()
    plt.savefig(SAVEDIR + "/pomdp_summary_by_env.pdf", bbox_inches="tight")

if __name__ == "__main__":
    main()