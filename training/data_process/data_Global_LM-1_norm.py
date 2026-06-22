"""Normalize ERA5 NetCDF files into model-ready 1-degree NumPy arrays."""

import os
import json
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
import logging

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ERA5_NC_ROOT = os.environ.get("YANTIAN_ERA5_NC_ROOT", "/home/dataset/ERA5-Global-LM")
ERA5_NORM_OUTPUT_ROOT = os.environ.get("YANTIAN_ERA5_NORM_OUTPUT_ROOT", "/home/dataset/ERA5-Global-LM-1-norm")

def setup_logger(start_year, end_year):
    logger = logging.getLogger('data_process_1_norm'+str(start_year)+str(end_year))
    logger.setLevel(logging.INFO)

    
    log_file_path = os.path.join(SCRIPT_DIR, 'data_process_1_norm'+str(start_year)+str(end_year)+'.log')
    file_handler = logging.FileHandler(log_file_path)

    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s') 
    file_handler.setFormatter(formatter)


    
    if not logger.handlers:
        logger.addHandler(file_handler)


    return logger

def load_avg_std():
    """Normalize ERA5 NetCDF files into model-ready 1-degree NumPy arrays."""
    avg_std_path = os.path.join(SCRIPT_DIR, 'statistics.json')
    with open(avg_std_path, 'r') as file:
        json_data = json.load(file)
    avg_list = json_data['avg']
    std_list = json_data['std']
    pressure_avg = np.array(avg_list[7:-13])
    pressure_std = np.array(std_list[7:-13])
    surface_avg_1 = avg_list[0:2]
    surface_avg_2 = avg_list[3:5]
    surface_avg = np.concatenate([surface_avg_1, surface_avg_2], axis=0)
    surface_std_1 = std_list[0:2]
    surface_std_2 = std_list[3:5]
    surface_std = np.concatenate([surface_std_1, surface_std_2], axis=0)
    avg = np.concatenate([pressure_avg, surface_avg], axis=0)
    std = np.concatenate([pressure_std, surface_std], axis=0)
    
    avg = np.expand_dims(avg, axis=(1,2))
    std = np.expand_dims(std, axis=(1,2))

    return avg, std
        




def get_file_path(dt, file_type):
    """Normalize ERA5 NetCDF files into model-ready 1-degree NumPy arrays."""
    year_dir = os.path.join(ERA5_NC_ROOT, str(dt.year))
    date_dir = os.path.join(year_dir, dt.strftime('%Y%m%d'))
    if file_type == 'pressure':
        
        filename = f'ERA5_Global_LM_Pressure_{dt.strftime("%Y%m%d%H")}.nc' 
    else:  # single
        
        filename = f'ERA5_Global_LM_Single_{dt.strftime("%Y%m%d%H")}.nc' 
    return os.path.join(date_dir, filename)

def _generate_time_indices(start_year, end_year):
    """Normalize ERA5 NetCDF files into model-ready 1-degree NumPy arrays."""
    time_indices = []
    
    for year in range(start_year, end_year + 1):
        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31, 18)
        
        current_date = start_date
        while current_date <= end_date:
            time_indices.append(current_date)
            current_date += timedelta(hours=6)
    
    return sorted(time_indices)  

def _load_data_for_time(dt, avg, std, logger):
    """Normalize ERA5 NetCDF files into model-ready 1-degree NumPy arrays."""
    upper_vars = ['z', 'r', 't', 'u', 'v']
    surface_vars = ['u10', 'v10', 't2m', 'msl']
    try:
        
        pressure_file = get_file_path(dt, 'pressure')
        upper_data_list = []
        with xr.open_dataset(pressure_file) as ds:
            for var in upper_vars:
                
                data = ds[var][:].astype(np.float32) 
                downsampled = data.coarsen(latitude=4, longitude=4, boundary="trim").mean()
                upper_data_list.append(downsampled)
        
        upper_data = np.concatenate(upper_data_list, axis=1).squeeze(0) 


        
        single_file = get_file_path(dt, 'single')
        surface_data_list = []
        with xr.open_dataset(single_file) as ds:
            for var in surface_vars:
                
                data = ds[var][:].astype(np.float32) 
                downsampled = data.coarsen(latitude=4, longitude=4, boundary="trim").mean()
                surface_data_list.append(downsampled)
        
        surface_data = np.concatenate(surface_data_list, axis=0) 

        
        data = np.concatenate([upper_data, surface_data], axis=0) 

        
        data = (data - avg) / std
        
        nan_count = np.isnan(data).sum()
        inf_count = np.isinf(data).sum()
        if nan_count > 0 or inf_count > 0:
            logger.info(f"Invalid values detected - NaN: {nan_count}, Inf: {inf_count}")
            
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            logger.info("Invalid values have been replaced with 0.")
        
        
        data = data.astype(np.float16)
        if dt == datetime(1987, 1, 1, 0, 0):
            logger.info(f"mean: {np.mean(data)}")
            logger.info(f"std: {np.std(data)}")
        
        
        
        year_dir = os.path.join(ERA5_NORM_OUTPUT_ROOT, str(dt.year))
        date_dir = os.path.join(year_dir, dt.strftime('%Y%m%d'))
        
        os.makedirs(date_dir, exist_ok=True)
        
        npy_path = os.path.join(date_dir, f"ERA5_Global_LM_{dt.strftime('%Y%m%d%H')}.npy")
        np.save(npy_path, data)
        
    except FileNotFoundError:
        logger.info(f"Data file not found for time {dt.strftime('%Y%m%d%H')}")
        
        
        
        return None 
    except KeyError:
        logger.info(f"Array named 'data' was not found for time {dt.strftime('%Y%m%d%H')}")
        # raise
        return None
    except Exception as e:
        logger.info(f"Error loading data for time {dt.strftime('%Y%m%d%H')}: {e}")
        # raise
        return None
 
def main():
    import time
    start_year = 1979
    end_year = 2022
    avg, std = load_avg_std()
    logger = setup_logger(start_year, end_year)
    time_indices = _generate_time_indices(start_year, end_year)
    for dt in time_indices:
        logger.info(f"--------Processing time: {dt.strftime('%Y%m%d%H')}--------")
        start_time = time.time()
        _load_data_for_time(dt, avg, std, logger)
        end_time = time.time()
        logger.info(f"Elapsed time: {end_time - start_time} seconds")
    logger.info("Data processing complete.")

if __name__ == "__main__":
    main()
