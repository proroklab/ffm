"""Plots a summary of MMER ordered by environment"""

import plotlib as pl
import matplotlib.pyplot as plt
import seaborn as sb


WORKDIR = "/Users/smorad/data/laplace"
SAVEDIR = WORKDIR + "/plots"


'''
projects = {"popgym-public": [15e6, 228], "FFM_4b2": [15e6, 228]}
runs, summary = pl.build_projects(projects, WORKDIR)
runs = runs.rename(columns={"MMER": "Max Cumulative Reward"})
# Stupid RLlib sometimes drops the last epochs
runs = runs[runs["Epoch"] <= 220]
sb.set()
sb.set_style("darkgrid")
sb.set_context('notebook')
ax = sb.relplot(runs, x="Epoch", y="Max Cumulative Reward", hue="Model", col="Env", kind='line', col_wrap=5, palette="tab20", facet_kws={'sharey': False, 'sharex': True})
sb.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

plt.tight_layout()
plt.savefig(SAVEDIR + "/lineplots.pdf", bbox_inches="tight")
'''
# Compute relative to mean of other baselines
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
runs, summary = pl.build_projects(projects, WORKDIR, clean=False, process_fn=pl.process_run_pomdp,
x_key="z/rl_steps",
metric_keys={
    "metrics/return_eval_total": "Episodic Reward",
    "z/rl_steps": "Env Steps",
},
recategorize_keys=["Env", "Model"],
)
runs = runs.replace(env_names)
runs = runs.replace({"GRU": "GRU/LSTM"})
runs = runs.rename(columns={"Reward": "Max Cumulative Reward"})
sb.set()
sb.set_style("darkgrid")
sb.set_context('notebook')
#breakpoint()
runs = runs.groupby(["Env", 'run_id', 'Model']).ewm(span=50).mean()

ax = sb.relplot(runs, x="Env Steps", y="Max Cumulative Reward", hue="Model", col="Env", kind='line', col_wrap=4, units="run_id", estimator=None, hue_order=["GRU/LSTM", "FFM"])
sb.move_legend(ax, "lower left", bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig(SAVEDIR + "/pomdp_sac_lineplots.pdf", bbox_inches="tight")

projects = {"FFM_extra_pomdp": [1_500_000, 0]}
runs, summary = pl.build_projects(projects, WORKDIR, clean=False, process_fn=pl.process_run_pomdp,
x_key="z/rl_steps",
metric_keys={
    "metrics/return_eval_total": "Episodic Reward",
    "z/rl_steps": "Env Steps",
},
recategorize_keys=["Env", "Model"],
)
runs = runs.replace(env_names)
runs = runs.replace({"GRU": "GRU/LSTM"})
runs = runs.rename(columns={"Reward": "Max Cumulative Reward"})
sb.set()
sb.set_style("darkgrid")
sb.set_context('notebook')
#breakpoint()
runs = runs.groupby(["Env", 'run_id', 'Model']).ewm(span=50).mean()

ax = sb.relplot(runs, x="Env Steps", y="Max Cumulative Reward", hue="Model", col="Env", kind='line', col_wrap=4, units="run_id", estimator=None, hue_order=["GRU/LSTM", "FFM"])
sb.move_legend(ax, "lower left", bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig(SAVEDIR + "/pomdp_td3_lineplots.pdf", bbox_inches="tight")