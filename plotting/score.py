
from plotlib import *
import matplotlib.pyplot as plt
import plotnine as p9
import seaborn as sb
import os

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
    projects = {"popgym-public": [15e6, 228], "FFM_4b2": [15e6, 228]}
    runs, summary = build_projects(projects, WORKDIR, clean=False)
    print(summary.groupby(["Model", "Env"]).mean().groupby("Model").mean()["MMER"])
    #nonav = summary[~summary["Env"].str.contains("Labyrinth")]
    #print(nonav.groupby(["Model", "Env"]).mean().groupby("Model").mean()["MMER"])


if __name__ == "__main__":
    main()