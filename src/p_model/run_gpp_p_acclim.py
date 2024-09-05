#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
calculate rolling mean of noon data and run P-model with acclimation and moisture stress
ref: https://doi.org/10.1029/2021MS002767; https://github.com/GiuliaMengoli/P-model_subDaily

author: rde, skoirala
first created: 2023-11-07
"""
import bottleneck as bn
import numpy as np

from src.common.get_params import get_params
from src.p_model.gpp_p_acclim import gpp_p_acclim


def getrollingmean(data, time_info, acclim_window):
    """
    calculate rolling mean of forcing variables for acclimation period

    parameters:
    data (array): forcing variable data
    time_info (dict): dictionary with time info
    acclim_window (int): acclimation window in days

    returns:
    data_acclim (array): rolling mean of forcing variables for acclimation period
    """
    window_size = int(np.floor(acclim_window))  # make acclimation window an integer
    # select noon data based on timeinfo and take average
    data_day = bn.nanmean(
        data.reshape(-1, time_info["nstepsday"])[
            :, time_info["start_hour_ind"] : time_info["end_hour_ind"]
        ],
        axis=1,
    )
    data_acclim = bn.move_mean(
        data_day, window=window_size, min_count=1
    )  # calculate rolling mean of average noon data
    return data_acclim


def get_daily_acclim_data(
    ip_df_dict, time_info, fpar_var_name, co2_var_name, acclim_window
):
    """
    select required forcing variables and calculate rolling mean for acclimation period

    parameters:
    ip_df_dict (dict): dictionary with input forcing data
    time_info (dict): dictionary with time info
    fpar_var_name (str): name of fpar variable
    co2_var_name (str): name of co2 variable
    acclim_window (int): acclimation window in days

    returns:
    data_sd (dict): dictionary with rolling mean of forcing variables for acclimation period
    """
    varibs = (
        "PPFD_IN_GF",
        fpar_var_name,
        "TA_GF",
        co2_var_name,
        "VPD_GF",
        "SW_IN_POT_ONEFlux",
        "Iabs",
    )  # forcing variables needed for PModelPlus
    data_dd = {}
    for _var in varibs:
        data_dd[_var] = getrollingmean(
            ip_df_dict[_var], time_info, acclim_window
        )  # calculate rolling mean of each forcing variables for acclimation period
    data_dd["elev"] = float(
        ip_df_dict["elev"]
    )  # Append elevation with daily (noon) data

    return data_dd


def run_p_model(
    p_values_scalar,
    p_names,
    ip_df_dict,
    model_op_no_acclim_sd,
    ip_df_daily_wai,
    wai_output,
    time_info,
    fpar_var_name,
    co2_var_name,
):
    """
    forward run PModel with acclimation and return GPP scaled by soil moisture stress

    parameters:
    p_values_scalar (array): array of scalar values for parameters
    p_names (list): list of parameter names
    ip_df_dict (dict): dictionary with input forcing data
    model_op_no_acclim_sd (dict): dictionary with PModel output without acclimation
    ip_df_daily_wai (dict): dictionary with input data for daily WAI calculation
    wai_output (dict): dictionary to store WAI output
    time_info (dict): dictionary with time info
    fpar_var_name (str): name of fpar variable
    co2_var_name (str): name of co2 variable

    returns:
    p_model_acclim_fw_op (dict): dictionary with PModel output with acclimation
    """

    # recalculate parameter values based on the scalar values given by optimizer
    updated_params = get_params(ip_df_dict, p_names, p_values_scalar)

    ip_df_daily_dict = get_daily_acclim_data(
        ip_df_dict,
        time_info,
        fpar_var_name,
        co2_var_name,
        updated_params["acclim_window"],
    )

    p_model_acclim_fw_op = gpp_p_acclim(
        ip_df_dict,
        ip_df_daily_dict,
        ip_df_daily_wai,
        wai_output,
        model_op_no_acclim_sd,
        time_info["nstepsday"],
        updated_params,
        co2_var_name,
    )

    return p_model_acclim_fw_op
