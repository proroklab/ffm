"""Plots a summary of MMER ordered by environment"""

import plotlib as pl
import matplotlib.pyplot as plt
import seaborn as sb
import os
from matplotlib import ticker 


WORKDIR = "/Users/smorad/data/laplace"
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
    projects = {"FFM_pomdp": [1_500_000, 0]}
    runs, summary = pl.build_projects(projects, WORKDIR, clean=False, process_fn=pl.process_run_pomdp, multiprocess=False, 
    x_key="z/rl_steps",
    metric_keys={
        "metrics/return_eval_total": "Episodic Reward",
        "z/rl_steps": "Env Steps",
    },
    recategorize_keys=["Env", "Model"],
    )
    summary = summary.replace(env_names)
    summary = summary.replace({"GRU": "GRU/LSTM"})

    # Plotting
    sb.set()
    sb.set_style("darkgrid")
    sb.set_context('paper')

    fig0, ax0 = plt.subplots(figsize=(10, 2))
    # TODO: use pointplot
    #order = summary.groupby('Env')['Reward'].mean()
    order = summary.groupby("Env").mean(numeric_only=True).sort_values("Reward").index
    sb.barplot(data=summary, x="Env", y="Reward", hue="Model", ax=ax0, order=order, hue_order=["GRU/LSTM", "FFM"])
    plt.tight_layout()
    plt.savefig(SAVEDIR + "/pomdp_summary_by_env.pdf", bbox_inches="tight")

if __name__ == "__main__":
    main()