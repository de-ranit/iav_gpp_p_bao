#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot boxplots of NSE from different optimization experiments of P Model
and LUE Model per PFT

Note:
Always run this script, after `cd` to the `prep_figs` directory
as the paths of result files are relative to this directory. The 
`prep_figs` directory should be a sub-directory of the main project directory.

author: rde
first created: Thu Feb 01 2024 16:25:46 CET
"""

import os
from pathlib import Path
import importlib
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib
import seaborn as sns

# set up matplotlib to use LaTeX for rendering text
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["text.latex.preamble"] = (
    r"\renewcommand{\familydefault}{\sfdefault}"  # use sans-serif font
)
matplotlib.rcParams["text.latex.preamble"] = (
    r"\usepackage{amsmath}"  # use amsmath
)

# set the font to computer modern sans
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = "cmss10"
plt.rcParams["pdf.fonttype"] = 42  # embedd fonts in pdf
plt.rcParams["axes.edgecolor"] = "black"  # make the axes edge color black
plt.rcParams["axes.linewidth"] = 2.0  # make the axes edge linewidth thicker


def prep_data(res_path_dict):
    """
    prepare data to plot boxplots of NSE from different optimization
    experiments of P Model and LUE Model per PFT

    Parameters:
    -----------
    res_path_dict (dict) : dict containing paths to the serialized model
    results from different optimization experiments of a certain model

    Returns:
    --------
    pft_mod_perform_dict (dict) : dict containing NSE from different
    optimization experiments of a certain Model per PFT
    """

    # find all the files with serialized model results
    mod_res_file_list = glob.glob(f"{res_path_dict['per_site_yr']}/*.npy")
    mod_res_file_list.sort()  # sort the files by site ID

    # filter out bad sites
    filtered_mod_res_file_list = [
        files
        for files in mod_res_file_list
        if not (
            "CG-Tch" in files
            or "MY-PSO" in files
            or "GH-Ank" in files
            or "US-LWW" in files
        )
    ]

    # create a dict to store the results from each PFT and each optimization
    # experiment
    pft_mod_perform_dict = {
        "CRO": {"per_site_yr": [], "per_site": [], "per_pft": [], "glob": []},
        "CSH": {"per_site_yr": [], "per_site": [], "per_pft": [], "glob": []},
        "DBF": {"per_site_yr": [], "per_site": [], "per_pft": [], "glob": []},
        "DNF": {"per_site_yr": [], "per_site": [], "per_pft": [], "glob": []},
        "EBF": {"per_site_yr": [], "per_site": [], "per_pft": [], "glob": []},
        "ENF": {"per_site_yr": [], "per_site": [], "per_pft": [], "glob": []},
        "GRA": {"per_site_yr": [], "per_site": [], "per_pft": [], "glob": []},
        "MF": {"per_site_yr": [], "per_site": [], "per_pft": [], "glob": []},
        "OSH": {"per_site_yr": [], "per_site": [], "per_pft": [], "glob": []},
        "SAV": {"per_site_yr": [], "per_site": [], "per_pft": [], "glob": []},
        "SNO": {"per_site_yr": [], "per_site": [], "per_pft": [], "glob": []},
        "WET": {"per_site_yr": [], "per_site": [], "per_pft": [], "glob": []},
        "WSA": {"per_site_yr": [], "per_site": [], "per_pft": [], "glob": []},
    }

    # for each site
    for res_file in filtered_mod_res_file_list:
        # open the results file from per site year optimization
        res_dict_site_yr = np.load(res_file, allow_pickle=True).item()
        site_id = res_dict_site_yr["SiteID"]

        # open the results file from other experiments
        res_dict_site = np.load(
            f"{res_path_dict['per_site']}/{site_id}_result.npy", allow_pickle=True
        ).item()
        res_dict_pft = np.load(
            f"{res_path_dict['per_pft']}/{site_id}_result.npy", allow_pickle=True
        ).item()
        res_dict_glob = np.load(
            f"{res_path_dict['glob_opti']}/{site_id}_result.npy", allow_pickle=True
        ).item()

        # get the site PFT
        pft = res_dict_site_yr["PFT"]

        # store NSE from different optimization experiments of a certain
        # Model based on the PFT of the site
        pft_mod_perform_dict[pft]["per_site_yr"].append(
            res_dict_site_yr["NSE"][f"NSE_{res_dict_site_yr['Temp_res']}"]
        )
        pft_mod_perform_dict[pft]["per_site"].append(
            res_dict_site["NSE"][f"NSE_{res_dict_site['Temp_res']}"]
        )
        pft_mod_perform_dict[pft]["per_pft"].append(
            res_dict_pft["NSE"][f"NSE_{res_dict_pft['Temp_res']}"]
        )
        pft_mod_perform_dict[pft]["glob"].append(
            res_dict_glob["NSE"][f"NSE_{res_dict_glob['Temp_res']}"]
        )

    return pft_mod_perform_dict


def plot_axs(ax, data_dict, title):
    """
    prepare the subplots for the figure

    Parameters:
    -----------
    ax (matplotlib.axes._subplots.AxesSubplot) : axes to plot the boxplots
    data_dict (dict) : dict containing NSE from different optimization
    experiments of a certain Model per PFT
    title (str) : title of the subplot

    Returns:
    --------
    None
    """

    # list of PFT
    pft_list = list(data_dict.keys())

    # colors for the boxplots (of each optimization experiment)
    colors = ["#56B4E9", "#009E73", "#D55E00", "#CC79A7"]

    # create boxplots
    for ix, pft_name in enumerate(pft_list):
        for jx, opti_type in enumerate(["per_site_yr", "per_site", "per_pft", "glob"]):
            # position for the boxplot
            position = 4 * ix + jx + 1
            # create the boxplot with specific color
            ax.boxplot(  # box_dict =
                1.0 / (2.0 - np.array(data_dict[pft_name][opti_type])),
                widths=0.5,
                positions=[position],
                patch_artist=True,
                boxprops=dict(facecolor=colors[jx]),
            )
            # for line in box_dict['medians']:
            #     # Get the median value
            #     median_value = line.get_ydata()[0]

            #     # Get the x position for the annotation
            #     x_position = line.get_xdata()[1]

            #     # Get the y position for the annotation
            #     y_position = median_value

            #     # Place the text annotation on the plot
            #     ax.text(x_position, y_position, f'{median_value:.2f}',
            #             ha='center', va='bottom', fontsize=8, color='blue')
            #     coll_dict[opti_type].append(round(median_value,2))

    # set x-axis labels as PFT (with number of sites in each PFT in brackets)
    ax.set_xticks(np.arange(2.5, 4 * len(pft_list) + 1, 4))
    xticklabs = [
        f"{pft_name} ({len(data_dict[pft_name]['per_site_yr'])})"
        for pft_name in pft_list
    ]
    ax.set_xticklabels(xticklabs)

    grid_x = np.arange(4.5,52.0,4)
    for i in grid_x:
        ax.axvline(i, color="gray", linestyle=(0, (5, 10)))


    # labeling axes
    ax.set_yticks(list(np.arange(0.0, 1.2, 0.2)))
    yticklabs = [round(n, 1) for n in list(np.arange(0.0, 1.2, 0.2))]
    ax.set_yticklabels(yticklabs)

    ax.tick_params(axis="both", which="major", labelsize=38)
    
    ax.set_title(
        title,
        fontsize=54,
    )

    sns.despine(ax=ax, top=True, right=True)


def plot_fig(p_mod_res_path_dict, lue_mod_res_path_dict):
    """
    plot the figure

    Parameters:
    -----------
    p_mod_res_path_dict (dict) : dict containing paths to the serialized model
    results from different optimization experiments of P Model
    lue_mod_res_path_dict (dict) : dict containing paths to the serialized model
    results from different optimization experiments of LUE Model

    Returns:
    --------
    None
    """
    # prepare the data for each model
    p_mod_pft_mod_perform_dict = prep_data(p_mod_res_path_dict)
    lue_mod_pft_mod_perform_dict = prep_data(lue_mod_res_path_dict)

    # prepare the figure
    fig, axs = plt.subplots(
        ncols=1, nrows=2, figsize=(40, 24), sharex=True, sharey=True
    )

    plot_axs(axs[0], p_mod_pft_mod_perform_dict, r"(a) P$^{\text{W}}_{\text{hr}}$ model")
    plot_axs(axs[1], lue_mod_pft_mod_perform_dict, r"(b) Bao$_{\text{hr}}$ model")

    fig.subplots_adjust(hspace=0.2)

    # Adding legend manually
    colors = ["#56B4E9", "#009E73", "#D55E00", "#CC79A7"]
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            label=opti_type,
            markerfacecolor=colors[i],
            markersize=30,
        )
        for i, opti_type in enumerate(
            [
                r"per site--year optimization",
                "per site optimization",
                "per PFT optimization",
                "global optimization",
            ]
        )
    ]

    plt.legend(
        handles=legend_elements,
        fontsize=40,
        loc="lower center",
        ncol=len(legend_elements),
        frameon=True,
        bbox_to_anchor=(0.5, -0.35),
    )

    fig.supxlabel("PFT", y=0.05, fontsize=54)
    fig.supylabel("NNSE [-]", x=0.08, fontsize=54)

    # save the figure
    fig_path = Path("figures")
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig("./figures/f09.png", dpi=300, bbox_inches="tight")
    plt.savefig("./figures/f09.pdf", dpi=300, bbox_inches="tight")
    plt.close("all")


if __name__ == "__main__":
    # get the result paths collection module
    result_paths = importlib.import_module("result_path_coll")

    # store all the paths in a dict (for p model)
    hr_p_model_res_path_coll = {
        "per_site_yr": result_paths.per_site_yr_p_model_res_path,
        "per_site": result_paths.per_site_p_model_res_path,
        "per_pft": result_paths.per_pft_p_model_res_path,
        "glob_opti": result_paths.glob_opti_p_model_res_path,
    }

    # store all the paths in a dict (for lue model)
    hr_lue_model_res_path_coll = {
        "per_site_yr": result_paths.per_site_yr_lue_model_res_path,
        "per_site": result_paths.per_site_lue_model_res_path,
        "per_pft": result_paths.per_pft_lue_model_res_path,
        "glob_opti": result_paths.glob_opti_lue_model_res_path,
    }

    # plot the figure
    plot_fig(hr_p_model_res_path_coll, hr_lue_model_res_path_coll)
