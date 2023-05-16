import plotlib as pl
import wandb
import csv
import json
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import ticker 


a_keys = {f'agg_a{i}': f'agg_a{i}' for i in range(32)} 
w_keys = {f'agg_w{i}': f'agg_w{i}' for i in range(4)}
keys = {**a_keys, **w_keys}
WORKDIR = "/Users/smorad/data/laplace"
SAVEDIR = WORKDIR + "/plots"
sb.set(rc={'text.usetex' : True})
sb.set_style("darkgrid")
sb.set_context('paper')
sb.set(font_scale=1.5)


def repeat_prev(runs):
    #runs = runs[runs['run_id'] == '3a282_00000']
    runs = runs[runs['run_id'] == '0318d_00000']
    #runs = runs[runs['run_id'].isin(['0318d_00000', '0318d_00001', '0318d_00002'])]

    plt.figure(figsize=(6, 4))
    #ax = sb.histplot(
    #    runs, x="Epoch", y=r"$t_\omega$", bins=32,
    #)
    ax = sb.lineplot(
        runs, x="Epoch", y=r"$t_\omega$", estimator=None, units='w_index',
    )
    #ax.arrow(215, 150, 0, -35, color="red", width=2.0)
    #ax.arrow(215, 30, 0, 18, color="red", width=2.0)
    ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, _: fr"$\geq$ {x:.0f}" if x >= 400 else f"{x:.0f}"))
    act, crit = 64, 104
    #yticks = [*ax.get_yticks(), act, crit]
    #yticklabels = [*ax.get_yticklabels(), act, crit]
    #ax.set_yticks(yticks, labels=yticklabels)
    ax.axhline(act, color="tab:purple", linewidth=3, label=f"Actor Mode($t={act}$)")
    ax.axhline(crit, color="tab:red", linewidth=3, label=f"Critic Mode($t={crit}$)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(SAVEDIR + "/repeat_prev_w.pdf", bbox_inches="tight")
    plt.figure(figsize=(6, 4))
    #ax = sb.histplot(
    #    runs, x="Epoch", y=r"$t_\alpha$", bins=32,
        #binrange=((0, 228), (0, 250))
    #)
    ax = sb.lineplot(
        runs, x="Epoch", y=r"$t_\alpha$", estimator=None, units='a_index', zorder=2,
    )
    act, crit = 32, 104
    ax.axhline(act, color="tab:purple", linewidth=3, zorder=1, label=f"Actor Mode($t={act})$")
    ax.axhline(crit, color="tab:red", linewidth=3, zorder=1, label=fr"Critic Mode($t \geq {crit})$")
    plt.legend(loc="upper left")
    #ax.arrow(210, 150, 10, 250, color="red", width=2.0)
    #ax.arrow(210, 60, 0, -11, color="red", width=2.0)
    ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, _: fr"$\geq$ {x:.0f}" if x >= 400 else f"{x:.0f}"))
    plt.tight_layout()
    plt.savefig(SAVEDIR + "/repeat_prev_a.pdf", bbox_inches="tight")

def cartpole(runs):
    runs = runs[runs['run_id'] == '72b25_00002']
    plt.figure(figsize=(4, 4))
    ax = sb.histplot(
        runs, x="Epoch", y=r"$t_\omega$", bins=32, 
        binrange=((0, 228), (0, 250))
    )
    plt.tight_layout()
    plt.savefig(SAVEDIR + "/cartpole_w.pdf", bbox_inches="tight")
    plt.figure(figsize=(4, 4))
    ax = sb.histplot(
        runs, x="Epoch", y=r"$t_\alpha$", bins=32,
        binrange=((0, 228), (0, 250))
    )
    plt.tight_layout()
    plt.savefig(SAVEDIR + "/cartpole_a.pdf", bbox_inches="tight")
    

def repeat_prev_init():
    projects = {"FFM_discuss": [15e6, 228]}
    run_filter = lambda run: run.name in ["RepeatPreviousMedium-GRU", 'RepeatPreviousMedium-RayFFM_32_104', 'RepeatPreviousMedium-RayFFM']
    runs, _ = pl.build_projects(projects, WORKDIR, clean=True, multiprocess=False,
        run_filter=run_filter
    )
    runs['Model'] = runs['Model'].astype(str)
    #runs.loc[runs['name'] == 'RepeatPreviousMedium-RayFFM_32_104', 'Model'] = r'FFM $t_\alpha = 32, t_\gamma = 104$'
    runs.loc[runs['name'] == 'RepeatPreviousMedium-RayFFM_32_104', 'Model'] = r'FFM-32,104'
    runs.loc[runs['name'] == 'RepeatPreviousMedium-RayFFM', 'Model'] = r'FFM-1,1024'
    plt.figure(figsize=(6, 4))
    runs = runs.rename(columns={"MMER": "Reward"})
    ax = sb.lineplot(
        runs, x="Epoch", y="Reward", hue="Model", units="run_id", estimator=None, hue_order=["FFM-32,104", "FFM-1,1024", "GRU"]
    )
    plt.legend(loc="lower right", ncol=1)
    #plt.setp(ax.get_legend().get_texts(), fontsize='8') # for legend text
    #plt.setp(ax.get_legend().get_title(), fontsize='8') # for legend title
    #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    #sb.move_legend(ax, bbox_to_anchor=(1, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(SAVEDIR + "/repeat_prev_init.pdf", bbox_inches="tight")

def count_recall_init():
    projects = {"FFM_discuss": [15e6, 228]}
    run_filter = lambda run: run.name in ["CountRecallMedium-GRU", 'CountRecallMedium-RayFFM_32_104', 'CountRecallMedium-RayFFM']
    runs, _ = pl.build_projects(projects, WORKDIR, clean=True, multiprocess=False,
        run_filter=run_filter
    )
    runs['Model'] = runs['Model'].astype(str)
    #runs.loc[runs['name'] == 'CountRecallMedium-RayFFM_32_104', 'Model'] = r'FFM $t_\alpha = 32, t_\gamma = 104$'
    runs.loc[runs['name'] == 'CountRecallMedium-RayFFM_32_104', 'Model'] = r'FFM-32,104'
    plt.figure(figsize=(6, 6))
    ax = sb.lineplot(
        runs, x="Epoch", y="MMER", hue="Model", units="run_id", estimator=None,
    )
    #plt.setp(ax.get_legend().get_texts(), fontsize='8') # for legend text
    #plt.setp(ax.get_legend().get_title(), fontsize='8') # for legend title
    #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()
    plt.savefig(SAVEDIR + "/repeat_prev_init.pdf", bbox_inches="tight")


def main():
    projects = {"FFM_discuss": [15e6, 228]}
    run_filter = lambda run: "FFM" in run.name
    runs, summary = pl.build_projects(projects, WORKDIR, clean=True,
        metric_keys={
            "episode_reward_mean": "Episodic Reward",
            "training_iteration": "Epoch",
            **keys,
        },
        run_filter=run_filter
    )
    id_vars = ['timesteps_total', "Model", "Env", "Epoch", 'run_id', 'name']
    melt_a = runs.melt(id_vars=id_vars, value_vars=a_keys, value_name='a_value', var_name='a_index')
    melt_w = runs.melt(id_vars=id_vars, value_vars=w_keys, value_name=r'$t_\omega$', var_name='w_index')
    runs = melt_a.merge(melt_w, on=["Epoch", "Env", "Model", "run_id", "timesteps_total", 'name'])
    # # Add t0, the initialized a, w values too
    # for run in runs["run_id"].unique():
    #     init_a = runs[runs["run_id"] == run].copy()
    #     linspace = np.log(0.01) / np.linspace(1, 1024, 32)
    #     for i, key in enumerate(a_keys):
    #         init_a[key] = linspace[i]
    #beta = 0.2
    beta = 0.1
    runs[r'$t_\alpha$'] = np.clip(np.log(beta)  / np.clip(runs['a_value'], -100, -1e-8), 0, 400)
    runs[r'$t_\omega$'] = np.clip(runs[r'$t_\omega$'], 0, 400)
    repeat_prev(runs)
    #cartpole(runs)
    repeat_prev_init()


if __name__ == '__main__':
    main()

# repeat previous
