#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
get the AIC/ AICc values to compare mechanistic and semi-empirical modelling experiments

author: rde
first created: Thu Jul 04 2024 17:17:08 CEST
"""

import json
import glob
from pathlib import Path
import importlib
import numpy as np
import pandas as pd


def calc_aic(obs, pred, no_of_param, normalize=False, correct=False):
    """
    calculate Akaike Information Criterion (AIC) for a model
    ref: https://doi.org/10.1177/0049124104268644

    Multimodel Inference: Understanding AIC and BIC in Model Selection
    Kenneth P. Burnham and David R. Anderson
    Sociological Methods Research 2004; 33; 261

    parameters:
    obs (array): observed data
    pred (array): predicted data
    no_of_param (int): number of parameters in the model

    returns:
    aic (float): AIC value
    """
    n = len(obs)

    rss = np.sum((obs - pred) ** 2.0)

    if correct:  # correct when n/no_of_param < 40
        aic = (
            (n * np.log(rss / n))
            + (2.0 * no_of_param)
            + ((2.0 * no_of_param * (no_of_param + 1.0)) / (n - no_of_param - 1.0))
        )  # page 270 in the ref
    else:
        aic = n * np.log(rss / n) + 2.0 * no_of_param  # page 268 in the ref

    if normalize:
        aic = aic / n

    return aic, rss, n, no_of_param


def aic_analysis_global(res_path_coll, opti_paths_coll, temp_res):
    """
    calculate global AIC values, RSS, no of observations and no of parameters

    parameters:
    -----------
    res_path_coll (dict): collection of result paths
    opti_paths_coll (dict): collection of optimization param paths
    temp_res (str): temporal resolution of the data

    returns:
    --------
    global_aic_val (dict): global AIC values
    rss_val_dict (dict): global RSS values
    n_val_dict (dict): global no of observations
    k_val_dict (dict): global no of parameters

    """

    total_no_of_param_dict = {}
    global_aic_val = {}
    rss_val_dict = {}
    n_val_dict = {}
    k_val_dict = {}

    for key, path in opti_paths_coll.items():
        opti_dict_file_list = glob.glob(f"{path}/*.json")

        total_no_of_parm = 0.0
        for file in opti_dict_file_list:
            with open(file, "r", encoding="utf-8") as f:
                opti_dict = json.load(f)

                if not np.isnan(opti_dict["xbest"][0]):
                    total_no_of_parm += len(opti_dict["xbest"])

        total_no_of_param_dict[key] = total_no_of_parm

        mod_res_file_list = glob.glob(f"{res_path_coll[key]}/*.npy")
        mod_res_file_list.sort()

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

        obs = np.array([])
        sim = np.array([])

        for res_file in filtered_mod_res_file_list:
            result_dict = np.load(res_file, allow_pickle=True).item()

            if temp_res == "hourly":
                mask = result_dict[f"GPP_drop_idx_{result_dict['Temp_res']}"].astype(
                    bool
                )

                obs = np.concatenate(
                    (
                        obs,
                        result_dict[f"GPP_NT_{result_dict['Temp_res']}"][~mask],
                    )
                )

                sim = np.concatenate(
                    (
                        sim,
                        result_dict[f"GPP_sim_{result_dict['Temp_res']}"][~mask],
                    )
                )

                aic, rss, n, no_of_param = calc_aic(obs, sim, total_no_of_parm)
                global_aic_val[f"{key}_aic"] = aic
                rss_val_dict[f"{key}_rss"] = rss
                n_val_dict[f"{key}_n"] = n
                k_val_dict[f"{key}_no_of_param"] = no_of_param

            elif temp_res == "daily":
                mask = result_dict["good_gpp_d_idx"].astype(bool)

                obs = np.concatenate(
                    (
                        obs,
                        result_dict["GPP_NT_daily"][mask],
                    )
                )

                sim = np.concatenate(
                    (
                        sim,
                        result_dict["GPP_sim_daily"][mask],
                    )
                )

                aic, rss, n, no_of_param = calc_aic(obs, sim, total_no_of_parm)
                global_aic_val[f"{key}_aic"] = aic
                rss_val_dict[f"{key}_rss"] = rss
                n_val_dict[f"{key}_n"] = n
                k_val_dict[f"{key}_no_of_param"] = no_of_param

            elif temp_res == "monthly":
                mask = result_dict["good_gpp_m_idx"].astype(bool)

                obs = np.concatenate(
                    (
                        obs,
                        result_dict["GPP_NT_monthly"][mask],
                    )
                )

                sim = np.concatenate(
                    (
                        sim,
                        result_dict["GPP_sim_monthly"][mask],
                    )
                )

                aic, rss, n, no_of_param = calc_aic(
                    obs, sim, total_no_of_parm, correct=True
                )
                global_aic_val[f"{key}_aic"] = aic
                rss_val_dict[f"{key}_rss"] = rss
                n_val_dict[f"{key}_n"] = n
                k_val_dict[f"{key}_no_of_param"] = no_of_param

            elif temp_res == "yearly":
                mask = result_dict["good_gpp_y_idx"].astype(bool)

                obs = np.concatenate(
                    (
                        obs,
                        result_dict["GPP_NT_yearly"][mask],
                    )
                )

                sim = np.concatenate(
                    (
                        sim,
                        result_dict["GPP_sim_yearly"][mask],
                    )
                )

                aic, rss, n, no_of_param = calc_aic(
                    obs, sim, total_no_of_parm, correct=True
                )
                global_aic_val[f"{key}_aic"] = aic
                rss_val_dict[f"{key}_rss"] = rss
                n_val_dict[f"{key}_n"] = n
                k_val_dict[f"{key}_no_of_param"] = no_of_param

    return global_aic_val, rss_val_dict, n_val_dict, k_val_dict


def format_func(value):
    """
    format values to present nicely in table
    """

    return "${:.2f} \\times 10^{{{:d}}}$".format(
        value / 10 ** (int(np.floor(np.log10(abs(value))))),
        int(np.floor(np.log10(abs(value)))),
    )


if __name__ == "__main__":
    # get the result paths collection module
    result_paths = importlib.import_module("result_path_coll")

    # store all the paths in a dict (for p model)
    hr_p_model_res_path_coll = {
        "per_site_yr": result_paths.per_site_yr_p_model_res_path,
        "per_site": result_paths.per_site_p_model_res_path,
        "per_site_iav": result_paths.per_site_p_model_res_path_iav,
        "per_pft": result_paths.per_pft_p_model_res_path,
        "glob_opti": result_paths.glob_opti_p_model_res_path,
    }

    lue_model_hr_res_path_coll = {
        "per_site_yr": result_paths.per_site_yr_lue_model_res_path,
        "per_site": result_paths.per_site_lue_model_res_path,
        "per_site_iav": result_paths.per_site_lue_model_res_path_iav,
        "per_pft": result_paths.per_pft_lue_model_res_path,
        "glob_opti": result_paths.glob_opti_lue_model_res_path,
    }

    lue_model_dd_res_path_coll = {
        "per_site_yr": result_paths.per_site_yr_dd_lue_model_res_path,
        "per_site": result_paths.per_site_dd_lue_model_res_path,
        "per_site_iav": result_paths.per_site_dd_lue_model_res_path_iav,
        "per_pft": result_paths.per_pft_dd_lue_model_res_path,
        "glob_opti": result_paths.glob_opti_dd_lue_model_res_path,
    }

    # collect paths where optimization results were stored
    site_yr_opti_path_p = Path(
        "../opti_results/P_model/"
        "site_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_nnse_unc_my_first_exp/"
        "opti_dicts/"
    )
    site_opti_path_p = Path(
        "../opti_results/P_model/"
        "all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_nnse_unc_my_first_exp/"
        "opti_dicts/"
    )
    site_iav_opti_path_p = Path(
        "../opti_results/P_model/"
        "all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_nnse_unc_my_first_exp/"
        "opti_dicts/"
    )
    pft_opti_path_p = Path(
        "../opti_results/P_model/"
        "per_pft_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_nnse_unc_my_first_exp/"
        "opti_dicts/"
    )
    global_opti_path_p = Path(
        "../opti_results/P_model/"
        "global_opti_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_nnse_unc_my_first_exp/"
        "opti_dicts/"
    )
    opti_dict_path_p_coll = {
        "per_site_yr": site_yr_opti_path_p,
        "per_site": site_opti_path_p,
        "per_site_iav": site_iav_opti_path_p,
        "per_pft": pft_opti_path_p,
        "glob_opti": global_opti_path_p,
    }

    site_yr_opti_path_lue_hr = Path(
        "../opti_results/LUE_model/"
        "site_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_my_first_exp/"
        "opti_dicts/"
    )
    site_opti_path_lue_hr = Path(
        "../opti_results/LUE_model/"
        "all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_my_first_exp/"
        "opti_dicts/"
    )
    site_iav_opti_path_lue_hr = Path(
        "../opti_results/LUE_model/"
        "all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_my_first_exp/"
        "opti_dicts/"
    )
    pft_opti_path_lue_hr = Path(
        "../opti_results/LUE_model/"
        "per_pft_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_my_first_exp/"
        "opti_dicts/"
    )
    global_opti_path_lue_hr = Path(
        "../opti_results/LUE_model/"
        "global_opti_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_nominal_cost_lue_my_first_exp/"
        "opti_dicts/"
    )
    opti_dict_path_lue_hr_coll = {
        "per_site_yr": site_yr_opti_path_lue_hr,
        "per_site": site_opti_path_lue_hr,
        "per_site_iav": site_iav_opti_path_lue_hr,
        "per_pft": pft_opti_path_lue_hr,
        "glob_opti": global_opti_path_lue_hr,
    }

    site_yr_opti_path_lue_dd = Path(
        "../opti_results/LUE_model/"
        "site_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_"
        "nominal_cost_lue_my_first_exp/"
        "opti_dicts/"
    )
    site_opti_path_lue_dd = Path(
        "../opti_results/LUE_model/"
        "all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_"
        "nominal_cost_lue_my_first_exp/"
        "opti_dicts/"
    )
    site_iav_opti_path_lue_dd = Path(
        "../opti_results/LUE_model/"
        "all_year_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_"
        "nominal_cost_lue_my_first_exp/"
        "opti_dicts/"
    )
    pft_opti_path_lue_dd = Path(
        "../opti_results/LUE_model/"
        "per_pft_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_"
        "nominal_cost_lue_my_first_exp/"
        "opti_dicts/"
    )
    global_opti_path_lue_dd = Path(
        "../opti_results/LUE_model/"
        "global_opti_BRK15_FPAR_FLUXNET_EO_CO2_MLO_NOAA_"
        "nominal_cost_lue_my_first_exp/"
        "opti_dicts/"
    )
    opti_dict_path_lue_dd_coll = {
        "per_site_yr": site_yr_opti_path_lue_dd,
        "per_site": site_opti_path_lue_dd,
        "per_site_iav": site_iav_opti_path_lue_dd,
        "per_pft": pft_opti_path_lue_dd,
        "glob_opti": global_opti_path_lue_dd,
    }

    ###########################################################
    (
        hr_global_aic_dict_p_model,
        hr_global_rss_dict_p_model,
        hr_global_n_dict_p_model,
        hr_global_k_dict_p_model,
    ) = aic_analysis_global(
        hr_p_model_res_path_coll,
        opti_dict_path_p_coll,
        "hourly",
    )
    (
        hr_global_aic_dict_lue_hr_model,
        hr_global_rss_dict_lue_hr_model,
        hr_global_n_dict_lue_hr_model,
        hr_global_k_dict_lue_hr_model,
    ) = aic_analysis_global(
        lue_model_hr_res_path_coll, opti_dict_path_lue_hr_coll, "hourly"
    )

    (
        dd_global_aic_dict_p_model,
        dd_global_rss_dict_p_model,
        dd_global_n_dict_p_model,
        dd_global_k_dict_p_model,
    ) = aic_analysis_global(
        hr_p_model_res_path_coll,
        opti_dict_path_p_coll,
        "daily",
    )
    (
        dd_global_aic_dict_lue_hr_model,
        dd_global_rss_dict_lue_hr_model,
        dd_global_n_dict_lue_hr_model,
        dd_global_k_dict_lue_hr_model,
    ) = aic_analysis_global(
        lue_model_hr_res_path_coll, opti_dict_path_lue_hr_coll, "daily"
    )
    (
        dd_global_aic_dict_lue_dd_model,
        dd_global_rss_dict_lue_dd_model,
        dd_global_n_dict_lue_dd_model,
        dd_global_k_dict_lue_dd_model,
    ) = aic_analysis_global(
        lue_model_dd_res_path_coll, opti_dict_path_lue_dd_coll, "daily"
    )

    (
        mm_global_aic_dict_p_model,
        mm_global_rss_dict_p_model,
        mm_global_n_dict_p_model,
        mm_global_k_dict_p_model,
    ) = aic_analysis_global(
        hr_p_model_res_path_coll,
        opti_dict_path_p_coll,
        "monthly",
    )
    (
        mm_global_aic_dict_lue_hr_model,
        mm_global_rss_dict_lue_hr_model,
        mm_global_n_dict_lue_hr_model,
        mm_global_k_dict_lue_hr_model,
    ) = aic_analysis_global(
        lue_model_hr_res_path_coll,
        opti_dict_path_lue_hr_coll,
        "monthly",
    )
    (
        mm_global_aic_dict_lue_dd_model,
        mm_global_rss_dict_lue_dd_model,
        mm_global_n_dict_lue_dd_model,
        mm_global_k_dict_lue_dd_model,
    ) = aic_analysis_global(
        lue_model_dd_res_path_coll,
        opti_dict_path_lue_dd_coll,
        "monthly",
    )

    (
        yr_global_aic_dict_p_model,
        yr_global_rss_dict_p_model,
        yr_global_n_dict_p_model,
        yr_global_k_dict_p_model,
    ) = aic_analysis_global(
        hr_p_model_res_path_coll,
        opti_dict_path_p_coll,
        "yearly",
    )
    (
        yr_global_aic_dict_lue_hr_model,
        yr_global_rss_dict_lue_hr_model,
        yr_global_n_dict_lue_hr_model,
        yr_global_k_dict_lue_hr_model,
    ) = aic_analysis_global(
        lue_model_hr_res_path_coll,
        opti_dict_path_lue_hr_coll,
        "yearly",
    )
    (
        yr_global_aic_dict_lue_dd_model,
        yr_global_rss_dict_lue_dd_model,
        yr_global_n_dict_lue_dd_model,
        yr_global_k_dict_lue_dd_model,
    ) = aic_analysis_global(
        lue_model_dd_res_path_coll,
        opti_dict_path_lue_dd_coll,
        "yearly",
    )

    hr_aic_df = pd.DataFrame(
        [hr_global_aic_dict_p_model, hr_global_aic_dict_lue_hr_model]
    )
    dd_aic_df = pd.DataFrame(
        [
            dd_global_aic_dict_p_model,
            dd_global_aic_dict_lue_hr_model,
            dd_global_aic_dict_lue_dd_model,
        ]
    )
    mm_aic_df = pd.DataFrame(
        [
            mm_global_aic_dict_p_model,
            mm_global_aic_dict_lue_hr_model,
            mm_global_aic_dict_lue_dd_model,
        ]
    )
    yr_aic_df = pd.DataFrame(
        [
            yr_global_aic_dict_p_model,
            yr_global_aic_dict_lue_hr_model,
            yr_global_aic_dict_lue_dd_model,
        ]
    )

    hr_rss_df = pd.DataFrame(
        [hr_global_rss_dict_p_model, hr_global_rss_dict_lue_hr_model]
    )
    dd_rss_df = pd.DataFrame(
        [
            dd_global_rss_dict_p_model,
            dd_global_rss_dict_lue_hr_model,
            dd_global_rss_dict_lue_dd_model,
        ]
    )
    mm_rss_df = pd.DataFrame(
        [
            mm_global_rss_dict_p_model,
            mm_global_rss_dict_lue_hr_model,
            mm_global_rss_dict_lue_dd_model,
        ]
    )
    yr_rss_df = pd.DataFrame(
        [
            yr_global_rss_dict_p_model,
            yr_global_rss_dict_lue_hr_model,
            yr_global_rss_dict_lue_dd_model,
        ]
    )

    hr_n_df = pd.DataFrame([hr_global_n_dict_p_model, hr_global_n_dict_lue_hr_model])
    dd_n_df = pd.DataFrame(
        [
            dd_global_n_dict_p_model,
            dd_global_n_dict_lue_hr_model,
            dd_global_n_dict_lue_dd_model,
        ]
    )
    mm_n_df = pd.DataFrame(
        [
            mm_global_n_dict_p_model,
            mm_global_n_dict_lue_hr_model,
            mm_global_n_dict_lue_dd_model,
        ]
    )
    yr_n_df = pd.DataFrame(
        [
            yr_global_n_dict_p_model,
            yr_global_n_dict_lue_hr_model,
            yr_global_n_dict_lue_dd_model,
        ]
    )

    hr_k_df = pd.DataFrame([hr_global_k_dict_p_model, hr_global_k_dict_lue_hr_model])
    dd_k_df = pd.DataFrame(
        [
            dd_global_k_dict_p_model,
            dd_global_k_dict_lue_hr_model,
            dd_global_k_dict_lue_dd_model,
        ]
    )
    mm_k_df = pd.DataFrame(
        [
            mm_global_k_dict_p_model,
            mm_global_k_dict_lue_hr_model,
            mm_global_k_dict_lue_dd_model,
        ]
    )
    yr_k_df = pd.DataFrame(
        [
            yr_global_k_dict_p_model,
            yr_global_k_dict_lue_hr_model,
            yr_global_k_dict_lue_dd_model,
        ]
    )

    # ############################################################
    hr_aic_df = hr_aic_df.applymap(format_func)
    dd_aic_df = dd_aic_df.applymap(format_func)
    mm_aic_df = mm_aic_df.applymap(format_func)
    yr_aic_df = yr_aic_df.applymap(format_func)

    hr_rss_df = hr_rss_df.applymap(format_func)
    dd_rss_df = dd_rss_df.applymap(format_func)
    mm_rss_df = mm_rss_df.applymap(format_func)
    yr_rss_df = yr_rss_df.applymap(format_func)

    hr_n_df = hr_n_df.applymap(format_func)
    dd_n_df = dd_n_df.applymap(format_func)
    mm_n_df = mm_n_df.applymap(format_func)
    yr_n_df = yr_n_df.applymap(format_func)

    hr_k_df = hr_k_df.applymap(format_func)
    dd_k_df = dd_k_df.applymap(format_func)
    mm_k_df = mm_k_df.applymap(format_func)
    yr_k_df = yr_k_df.applymap(format_func)

    print("######################################")
    print("Hourly level AIC/rss/n/k values")
    print(hr_aic_df.to_latex(index=False, escape=False))
    print(hr_rss_df.to_latex(index=False, escape=False))
    print(hr_n_df.to_latex(index=False, escape=False))
    print(hr_k_df.to_latex(index=False, escape=False))

    print("######################################")
    print("Daily level AIC/rss/n/k values")
    print(dd_aic_df.to_latex(index=False, escape=False))
    print(dd_rss_df.to_latex(index=False, escape=False))
    print(dd_n_df.to_latex(index=False, escape=False))
    print(dd_k_df.to_latex(index=False, escape=False))

    print("######################################")
    print("Monthly level AICc/rss/n/k values")
    print(mm_aic_df.to_latex(index=False, escape=False))
    print(mm_rss_df.to_latex(index=False, escape=False))
    print(mm_n_df.to_latex(index=False, escape=False))
    print(mm_k_df.to_latex(index=False, escape=False))

    print("######################################")
    print("Yearly level AICc/rss/n/k values")
    print(yr_aic_df.to_latex(index=False, escape=False))
    print(yr_rss_df.to_latex(index=False, escape=False))
    print(yr_n_df.to_latex(index=False, escape=False))
    print(yr_k_df.to_latex(index=False, escape=False))
