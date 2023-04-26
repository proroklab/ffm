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

def percentage(base, other):
    return 100 * (other - base) / base

def main():
    projects = {"popgym-public": [15e6, 228], "FFM_4b2": [15e6, 228]}
    runs, summary = pl.build_projects(projects, WORKDIR, clean=False)
    print(summary[summary["Model"] == "FFM"].groupby("Env").mean().mean()["MMER"])
    summary = summary.set_index(["Model", "Env", "trial_idx"])
    ffm = summary.loc["FFM"]
    gru = summary.loc["GRU"]
    fart = summary.loc["FART"]
    other_models = [m for m in summary.index.get_level_values('Model').unique() if m != "FFM"]
    other = summary.loc[other_models]

    # We only want to see the variance of FFM, so mean all the others
    ffm.loc[:, "Reward Relative to Mean"] = percentage(other.groupby("Env")["Normalized MMER"].mean(), ffm["Normalized MMER"])
    ffm.loc[:,"Reward Relative to GRU"] = percentage(gru.groupby("Env")["Normalized MMER"].mean(), ffm["Normalized MMER"])
    ffm.loc[:,"Reward Relative to FART"] = percentage(fart.groupby("Env")["Normalized MMER"].mean(), ffm["Normalized MMER"])

    # Plotting
    sb.set()
    sb.set_style("darkgrid")
    sb.set_context('paper')

    fig0, ax0 = plt.subplots(figsize=(10, 1.8))
    # Reset index so we can use Env as x in seaborn
    ffm_vs_mean = ffm.reset_index()
    order = ffm_vs_mean.groupby("Env").mean(numeric_only=True).sort_values("Reward Relative to Mean").index
    sb.barplot(data=ffm_vs_mean, x="Env", y="Reward Relative to Mean", ax=ax0, order=order, color="cornflowerblue")
    ax0.set_xticklabels(ax0.get_xticklabels(), rotation=25, horizontalalignment='right', size=6)
    #ax0.yaxis.set_minor_formatter(ticker.ScalarFormatter())
    ax0.set(yscale="symlog")
    ax0.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:.0f}%'))


    plt.savefig(SAVEDIR + "/mmer_summary_by_env_mean.pdf", bbox_inches="tight")

    fig1, ax1 = plt.subplots(figsize=(10, 1.8))
    # Reset index so we can use Env as x in seaborn
    ffm_vs_gru = ffm.reset_index()
    order = ffm_vs_gru.groupby("Env").mean(numeric_only=True).sort_values("Reward Relative to GRU").index
    sb.barplot(data=ffm_vs_gru, x="Env", y="Reward Relative to GRU", ax=ax1, order=order, color="cornflowerblue")
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=25, horizontalalignment='right', size=6)
    ax1.set(yscale="symlog")
    ax1.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:.0f}%'))
    plt.savefig(SAVEDIR + "/mmer_summary_by_env_gru.pdf", bbox_inches="tight")

    fig2, ax2 = plt.subplots(figsize=(10, 1.8))
    # Reset index so we can use Env as x in seaborn
    ffm_vs_fart = ffm.reset_index()
    order = ffm_vs_fart.groupby("Env").mean(numeric_only=True).sort_values("Reward Relative to FART").index
    sb.barplot(data=ffm_vs_fart, x="Env", y="Reward Relative to FART", ax=ax2, order=order, color="cornflowerblue")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=25, horizontalalignment='right', size=6)
    ax2.set(yscale="symlog")
    ax2.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:.0f}%'))
    plt.savefig(SAVEDIR + "/mmer_summary_by_env_fart.pdf", bbox_inches="tight")

    fig3, ax3 = plt.subplots(figsize=(10, 1.8))
    top = summary.loc[(['FFM', 'GRU', 'FART'])]
    #order = ffm_vs_fart.groupby("Env").mean(numeric_only=True).sort_values("Reward Relative to FART").index
    order = top.reset_index().groupby("Env").mean(numeric_only=True).sort_values("MMER").index
    sb.pointplot(data=top.reset_index(), x="Env", y="Normalized MMER", hue="Model", ax=ax3, palette="Set2", order=order)
    #sb.barplot(data=ffm_all, x="Env", y="Normalized MMER", ax=ax3, color="cornflowerblue")
    #sb.barplot(data=gru_all, x="Env", y="Normalized MMER", ax=ax3, color="red")
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=35, horizontalalignment='right', size=6)
    plt.savefig(SAVEDIR + "/mmer_summary.pdf", bbox_inches="tight")

if __name__ == "__main__":
    main()