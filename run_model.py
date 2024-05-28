import numpy as np
import netCDF4 as nc

from model_func import *
import os
from tqdm import tqdm
import time



#adjust these depending on your data 
years = np.arange(2000, 2024, 1)
num_days = 8766
params = [420, 8, 0.2, 0.8, 1.5, (0.75, 0.25)]
longitude_values = np.arange(22)
latitude_values = np.arange(22)
# create folder for results
os.makedirs('calibration_results', exist_ok=True)

full_data = np.zeros((len(latitude_values), len(longitude_values), num_days, 4))
'''full data array with dimensions [lat, lon, days, variable] will be filled in the following 
    code snipped, where variable equal to x -> y:
            0 -> P: precipitation data [m/day]
            1 -> R: net radiation data [J/day/m**2]
            2 -> T: temperature [K]
            3 -> LAI: leaf area index [m**2/m**2]'''
# need to adjust the file_path1 to the correct files, loops over all lattitude and longitude values
# as well as years and stores the data in the full_data array. Couldn't figure out a more efficient 
# way to do this
for i, lat in enumerate(latitude_values):
    for j, lon in tqdm(enumerate(longitude_values)):
        R_data = []
        P_data = []
        T_data = []
        lai_data = []
        # get radiation, temperature and precipitation data from netCDF files
        for year in years:
            file_path1 = 'data/total_precipitation/tp.daily.calc.era5.0d50_CentralEurope.' + str(
                year) + '.nc'
            nc_file = nc.Dataset(file_path1)
            # 7,8 is the grid cell of interest for the respective catchment area
            dates = nc_file.variables['time'][:]
            P_data.append(nc_file.variables['tp'][:, lat, lon])
            nc_file.close()
            file_path1 = 'data/net_radiation/nr.daily.calc.era5.0d50_CentralEurope.' + str(
                year) + '.nc'
            nc_file = nc.Dataset(file_path1)
            # print(nc_file)
            R_data.append(nc_file.variables['nr'][:, lat, lon])
            nc_file.close()
            file_path1 = 'data/daily_average_temperature/t2m_mean.daily.calc.era5.0d50_CentralEurope.' + str(
                year) + '.nc'
            nc_file = nc.Dataset(file_path1)
            T_data.append(nc_file.variables['t2m'][:, lat, lon])
            nc_file.close()
            file_path1 = 'data/lai/lai.daily.0d50_CentralEurope.' + str(
                year) + '.nc'
            nc_file = nc.Dataset(file_path1)
            lai_data.append(nc_file.variables['lai'][:, lat, lon])
            nc_file.close()
        full_data[i, j, :, 0] = np.concatenate(P_data)
        full_data[i, j, :, 1] = np.concatenate(R_data)
        full_data[i, j, :, 2] = np.concatenate(T_data)
        full_data[i, j, :, 3] = np.concatenate(lai_data)

# performs time evolution of SWBM with numpy arrays, makes the code way faster thant looping over each grid cell
results = time_evolution(full_data, *params) 
# convert results to nc format
res_xr_ds = out2xarray(results)

# Save results
out_path = f'results/model_output_{time.time()}'
os.makedirs(out_path, exist_ok=True)
for out in ['runoff', 'evapotranspiration', 'soil_moisture', 'snow']:
    res_xr_ds[out].to_netcdf(f'{out_path}/{out}.nc')
