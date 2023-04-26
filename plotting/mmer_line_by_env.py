"""Plots a summary of MMER ordered by environment"""

from plotlib import *
import matplotlib.pyplot as plt
import plotnine as p9
import patchworklib as pw


def percentage(base, other):
    return 100 * (other - base) / base


projects = {"popgym-public": [15e6, 228], "FFM_glu": [15e6, 228]}
runs, summary = build_projects(projects)
ffm = summary[summary["Model"] == "FFM"]
other = summary[summary["Model"] != "FFM"]
ffm["Relative MMER"] = percentage(ffm, other.groupby("Env").transform("mean"))
result = sort_by_cat(summary, sort_key="Relative MMER", value_key="Env")
# Compute relative to mean of other baselines
breakpoint()
(
    p9.ggplot(result)
    + p9.geom_col(p9.aes(x="Env", y="Relative MMER"))
    + p9.theme(axis_text_x=p9.element_text(rotation=70, hjust=1))
)
