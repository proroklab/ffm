import io
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import plotlib as pl
import numpy as np
import getpass


WORKDIR = f"/Users/{getpass.getuser()}/data/laplace"
SAVEDIR = WORKDIR + "/plots"
CSVPATH = WORKDIR + "/throughput.csv"

df = pd.read_csv(CSVPATH)
df["color"] = df["model"] == "RayFFM"
df = df.rename(columns={"model": "Model", "mem (MB)": "Peak Memory (MB)", "MMER": "Reward"})
df = df.replace(pl.model_renames)

sb.set(font_scale=1.5)

plt.figure(figsize=(5, 5))
train = df[df["mode"] == "train"]
mask = train[(train["Model"] == "FFM") & (train["device"] == "cpu")].index
train.loc[mask, "Model"] = "FFM (CPU)"
train = train.rename({"time (ms)": "Train Time (ms)"}, axis=1)
order = train.groupby('Model').mean(numeric_only=True).sort_values(by="Train Time (ms)").index
#train.sort_values(by="Train Time (ms)", inplace=True)
ax = sb.barplot(y="Model", x="Train Time (ms)", data=train, hue="color", dodge=False, order=order)
ax.set(ylabel=None)
ax.set(xscale="log")
ax.legend_.remove()
plt.tight_layout()
plt.savefig(SAVEDIR + "/train_time.pdf", bbox_inches="tight")
plt.clf()
#plt.show()

plt.figure(figsize=(5, 5))
cpu_inf = df[(df["mode"] == "inference") & (df["device"] == "cpu")]
cpu_inf = cpu_inf.rename({"time (ms)": "Inference Latency (ms)"}, axis=1)
#cpu_inf.sort_values(by="Inf. Latency (ms)", inplace=True)
order = cpu_inf.groupby('Model').mean(numeric_only=True).sort_values(by="Inference Latency (ms)").index
ax = sb.barplot(y="Model", x="Inference Latency (ms)", data=cpu_inf, hue="color", dodge=False, order=order)
ax.legend_.remove()
ax.set(ylabel=None)
plt.tight_layout()
plt.savefig(SAVEDIR + "/inference_time.pdf", bbox_inches="tight")
plt.clf()
#plt.show()

plt.figure(figsize=(5, 5))
mem = df[(df["mode"] == "train") & (df["device"] == "cuda")]
#mem = mem.sort_values(by="Peak Memory (MB)")
order = mem.groupby('Model').mean(numeric_only=True).sort_values(by="Peak Memory (MB)").index
ax = sb.barplot(y="Model", x="Peak Memory (MB)", data=mem, hue="color", dodge=False, order=order, errorbar=None)
ax.legend_.remove()
ax.set(ylabel=None)
plt.tight_layout()
plt.savefig(SAVEDIR + "/train_memory.pdf", bbox_inches="tight")
plt.clf()
#plt.show()


plt.figure(figsize=(5, 5))
param = df[(df["mode"] == "train") & (df["device"] == "cuda")]
#param = param.sort_values(by="num_params (K)")
order = param.groupby('Model').mean(numeric_only=True).sort_values(by="num_params (K)").index
param["# Params"] = param["num_params (K)"] * 1000
ax = sb.barplot(y="Model", x="# Params", data=param, hue="color", dodge=False, order=order)
ax.legend_.remove()
ax.set(ylabel=None)
plt.tight_layout()
plt.savefig(SAVEDIR + "/num_params.pdf", bbox_inches="tight")
plt.clf()
#plt.show()


AGG_MAP = {
    "Base Env": "first", 
    "Difficulty": "first", 
    "Model": "first",
    "Reward": "mean",
    "Normalized MMER": "mean",
}


plt.figure(figsize=(5, 5))
projects = {"popgym-public": [15e6, 228], "FFM_4b2": [15e6, 228]}
#projects = {"FFM_4b2": [15e6, 228]}
runs, summary = pl.build_projects(projects, WORKDIR, clean=False)
summary = summary.rename(columns={"model": "Model", "mem (MB)": "Peak Memory (MB)", "MMER": "Reward"})
summary = summary.replace(pl.model_renames)
mask = summary['trial_idx'] < 3
summary["color"] = summary["Model"] == "FFM"
res = summary[mask].groupby(['Model', 'trial_idx']).mean(numeric_only=True)
#res = summary.groupby(["Env", "Model"]).agg(AGG_MAP)
#res = summary.groupby(["Env", "Model"]).mean(numeric_only=True).groupby('Model').mean(numeric_only=True)
#res = res.sort_values(by="MMER").reset_index()
order = res.groupby('Model').mean(numeric_only=True).sort_values(by="Reward").index
ax = sb.barplot(y="Model", x="Reward", data=res.reset_index(), hue="color", dodge=False, order=order)
ax.legend_.remove()
ax.set(ylabel=None)
plt.tight_layout()
plt.savefig(SAVEDIR + "/mmer_by_model.pdf", bbox_inches="tight")
plt.clf()