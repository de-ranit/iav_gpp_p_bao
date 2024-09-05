#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module calculates the partial sensitivity functions and
simulates GPP using the LUE model

author: rde
first created: Tue Dec 26 2023 17:33:47 CET
"""

from src.common.get_params import get_params
from src.common.wai import calc_wai
from src.lue_model.partial_sensitivity_funcs import (
    f_temp_horn,
    f_vpd_co2_preles,
    f_water_horn,
    f_light_scalar_tal,
    f_cloud_index_exp,
)


def run_lue_model(
    param_values_scalar,
    param_names,
    ip_df_dict,
    ip_df_daily_wai,
    wai_output,
    nstepsday,
    fpar_var_name,
    co2_var_name,
):
    """
    Calculate partial sensitivity functions and simulate GPP using the LUE model

    Parameters:
    param_values_scalar (array): array of scalar of parameter values
    param_names (list): list of parameter names, which are optimized
    ip_df_dict (dict): dictionary with input forcing data
    ip_df_daily_wai (dict): dictionary with input data for daily WAI calculation
    wai_output (dict): dictionary to store WAI output
    nstepsday (int): number of sub-daily timesteps in a day
    fpar_var_name (str): name of fAPAR variable
    co2_var_name (str): name of CO2 variable

    Returns:
    lue_model_op (dict): dictionary with LUE model output
    """
    # recalculate parameter values based on the scalar values given by optimizer
    updated_params = get_params(ip_df_dict, param_names, param_values_scalar)

    # calculate WAI
    # first do spinup for calculation of wai using daily data
    # (for faster spinup loops when actual simulations are sub daily)
    wai_results = calc_wai(
        ip_df_daily_wai,
        wai_output,
        updated_params,
        wai0=updated_params["AWC"],
        nloops=updated_params["nloop_wai_spin"],
        spinup=True,
        do_snow=True,
        normalize_wai=False,
        do_sublimation=True,
        nstepsday=nstepsday,
    )

    # then when steady state is achieved (after 5 spinup loops),
    # do the actual calculation of wai using subdaily/daily data
    wai_results = calc_wai(
        ip_df_dict,
        wai_results,
        updated_params,
        wai0=wai_results["wai"][-1],
        nloops=updated_params["nloop_wai_act"],
        spinup=False,
        do_snow=True,
        normalize_wai=True,
        do_sublimation=True,
        nstepsday=nstepsday,
    )

    # calculate sensitivity function for temperature
    f_tair = f_temp_horn(
        ip_df_dict["TA_GF"],
        updated_params["T_opt"],
        updated_params["K_T"],
        updated_params["alpha_fT_Horn"],
    )

    # calculate sensitivity function for VPD
    f_vpd_co2, f_vpd_part, f_co2_part = f_vpd_co2_preles(
        ip_df_dict["VPD_GF"],
        ip_df_dict[co2_var_name],
        updated_params["Kappa_VPD"],
        updated_params["Ca_0"],
        updated_params["C_Kappa"],
        updated_params["c_m"],
    )

    # calculate sensitivity function for soil moisture
    f_water = f_water_horn(
        wai_results["wai_nor"],
        updated_params["W_I"],
        updated_params["K_W"],
        updated_params["alpha"],
    )

    # calculate sensitivity function for light scalar
    f_light = f_light_scalar_tal(
        ip_df_dict[fpar_var_name],
        ip_df_dict["PPFD_IN_GF"],
        updated_params["gamma_fL_TAL"],
    )

    # calculate sensitivity function for cloudiness index
    f_cloud, ci = f_cloud_index_exp(
        mu_fci=updated_params["mu_fCI"],
        sw_in=ip_df_dict["SW_IN_GF"],
        sw_in_pot=ip_df_dict["SW_IN_POT_ONEFlux"],
    )

    # calculate GPP
    gpp_lue = (
        updated_params["LUE_max"] * f_tair * f_vpd_co2 * f_water * f_light * f_cloud
    ) * (ip_df_dict["PPFD_IN_GF"] * ip_df_dict[fpar_var_name])

    # store model outputs in dictionary
    wai_results["fW"] = f_water
    lue_model_op = {}
    lue_model_op["gpp_lue"] = gpp_lue
    lue_model_op["fT"] = f_tair
    lue_model_op["fVPD"] = f_vpd_co2
    lue_model_op["fVPD_part"] = f_vpd_part
    lue_model_op["fCO2_part"] = f_co2_part
    lue_model_op["fW"] = f_water
    lue_model_op["fL"] = f_light
    lue_model_op["fCI"] = f_cloud
    lue_model_op["ci"] = ci
    lue_model_op["wai_results"] = wai_results

    return lue_model_op
