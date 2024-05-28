import numpy as np
import netCDF4 as nc

import pandas as pd
from scipy.stats import pearsonr
from tqdm import tqdm
import xarray as xr


def date_range(start, end):
    # Convert start and end dates to datetime64[ns] objects
    start_date = np.datetime64(start)
    end_date = np.datetime64(end)

    # Generate the date range as datetime64[ns] objects
    date_range = np.arange(start_date, end_date + np.timedelta64(1, 'D'),
                           dtype='datetime64[D]')

    return pd.to_datetime(date_range)


def calc_et_weight(temp, lai, w):
    """Calculate influence of LAI and temperature on ET.
    Input: temp: temperature data [K]
           lai: leaf area index data [m**2/m**2]
           w: weights for temperature and lai"""
    # Get coefficients for temperature and lai
    temp_w, lai_w = w
    lai = np.nan_to_num(lai, nan=0)
    temp_min = temp.min(axis=2, keepdims=True)
    temp_max = temp.max(axis=2, keepdims=True)
    lai_min = lai.min(axis=2, keepdims=True)
    lai_max = lai.max(axis=2, keepdims=True)

    # Perform normalization
    normalized_temp = (temp - temp_min) / (temp_max - temp_min)
    normalized_lai = (lai - lai_min) / (lai_max - lai_min)

    # Weight Temperature and LAI
    et_coef = temp_w * normalized_temp + lai_w * normalized_lai
    return et_coef


def runoff(wn, Pn, cs, alpha):
    return Pn * (wn / cs) ** alpha


def evapotranspiration(wn, Rn, cs, beta, gamma):
    return beta * (wn / cs) ** gamma * Rn


def snow_function(Snow_n, P_n, T_n, c_m):
    snow_stays_mask = T_n > np.ones_like(T_n) * 273.15
    no_snow_mask = Snow_n >= 0.001 * np.ones_like(T_n)

    # -- Calculate snow
    snow_masked = np.ma.array(Snow_n + P_n, mask=snow_stays_mask)
    snow_out_masked = np.ma.array(
        snow_masked.filled(fill_value=np.zeros_like(Snow_n)),
        mask=snow_stays_mask & no_snow_mask,
        fill_value=snow_masked.fill_value)

    # melting snow
    SnowMelt = c_m * (T_n - 273.15)

    snow_out = snow_out_masked.filled(fill_value=Snow_n - SnowMelt)

    snow_out[snow_out < 0] = 0

    # -- Calculate water
    water_masked = np.ma.array(np.zeros_like(Snow_n), mask=snow_stays_mask)
    water_out_masked = np.ma.array(
        water_masked.filled(fill_value=P_n),
        mask=snow_stays_mask & no_snow_mask,
        fill_value=snow_masked.fill_value)
    water_out = water_out_masked.filled(fill_value=SnowMelt + P_n)

    return snow_out, water_out


def water_balance(wn, Pn, Rn, Snown, Tn, cs, alpha, beta, gamma, c_m):
    """ Calculates the water balance for one time step as introduced in the lecture. Added features, such as snow"""
    snow, Pn = snow_function(Snown, Pn, Tn,
                             c_m)  # overwrites the precipitation (if snow melts or precipitation is accumulated as snow)
    Qn = runoff(wn, Pn, cs, alpha)
    En = evapotranspiration(wn, Rn, cs, beta, gamma)
    w_next = wn + (Pn - En - Qn)
    w_next = np.maximum(0, w_next)
    return Qn, En, w_next, snow


def out2xarray(output, start_year=2000, end_year=2024):
    """Converts output array into an nc file to save"""
    output = np.moveaxis(output, 2, 0)  # move time axis to be first axis

    # get dates and coordinates
    times = date_range('2000-01-01', '2023-12-31')
    nc_file = nc.Dataset(f'data/total_precipitation/tp.daily.calc.era5.0d50_CentralEurope.2005.nc')
    lons = nc_file.variables['lon'][:].data
    lats = nc_file.variables['lat'][:].data
    nc_file.close()

    out_dict = {}
    for i, out_name in enumerate(['runoff',
                                  'evapotranspiration',
                                  'soil_moisture',
                                  'snow']):
        out_xr = xr.DataArray(output[:, :, :, i], dims=('time', 'lat', 'lon'),
                              coords={'time': times,
                                      'lat': lats,
                                      'lon': lons})
        out_dict[out_name] = out_xr

    return xr.Dataset(out_dict)


def time_evolution(full_data, cs, alpha, gamma, beta, c_m, et_weight):
    """Calculates the time evolution of the soil moisture, runoff and evapotranspiration and snow.
    Input:  full data array with dimensions [lat, lon, days, variable] where variable equal to x -> y:
            0 -> P: precipitation data [m/day]
            1 -> R: net radiation data [J/day/m**2]
            2 -> T: temperature [K]
            3 -> LAI: leaf area index [m**2/m**2]
            cs: Soil water holding capacity [mm]
            alpha: runoff parameter
            beta: evapotranspiration parameter
            gamma: evapotranspiration parameter
            c_m: snow melt parameter [mm/K/day]
    Output: numpy array with dimensions [lat, lon, days, variable] where variable equal to x -> y:
            0 -> runoff [mm/day]
            1 -> evapotranspiration [mm/day]
            2 -> soil moisture [mm]
            3 -> snow [mm]"""
    # read out data from full_ data_frame
    P_data = full_data[:, :, :, 0]
    R_data = full_data[:, :, :, 1]
    T_data = full_data[:, :, :, 2]
    lai_data = full_data[:, :, :, 3]

    # convert units to ensure consitency
    conv = 1 / 2260000  # from J/day/m**2 to mm/day
    R_data = R_data * conv
    P_data = P_data * 10 ** 3  # from m/day to mm/day

    w_0 = 0.9 * cs * np.ones_like((P_data[:, :, 0]))
    Snow_0 = np.zeros_like((P_data[:, :, 0]))

    output = np.zeros(
        (len(P_data[:, 0, 0]), len(P_data[0, :, 0]), len(P_data[0, 0, :]), 4))

    # Precompute ET parameter
    et_coefs = beta * calc_et_weight(T_data, lai_data, et_weight)
    print('start_timeevolution')
    # Time evolution for all grid cells at the same time
    for t in tqdm(range(1, len(P_data[0, 0, :]) + 1)):

        P = P_data[:, :, t - 1]
        R = R_data[:, :, t - 1]
        T = T_data[:, :, t - 1]
        et_coef = et_coefs[:, :, t - 1]

        run_off, evapo, soil_mois, snow = water_balance(w_0, P, R, Snow_0, T, cs, alpha,
                                      et_coef, gamma, c_m)
        output[:, :, t - 1, 0] = run_off
        output[:, :, t - 1, 1] = evapo
        output[:, :, t - 1, 2] = soil_mois
        output[:, :, t - 1, 3] = snow

        w_0 = soil_mois
        Snow_0 = snow

    return output


