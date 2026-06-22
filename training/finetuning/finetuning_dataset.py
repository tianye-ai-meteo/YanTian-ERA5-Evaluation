"""Dataset utilities for multi-step fine-tuning on normalized ERA5 arrays."""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr
from datetime import datetime, timedelta
from config import *
import time

class BaselineDataset(Dataset):
    def __init__(self, root_dir=data_dir, start_year=1987, end_year=2016, step=6):
        """Dataset utilities for multi-step fine-tuning on normalized ERA5 arrays."""
        self.root_dir = root_dir
        self.start_year = start_year
        self.end_year = end_year
        self.step = step

        
        
        self.time_indices = self._generate_time_indices()
        
    def _generate_time_indices(self):
        """Dataset utilities for multi-step fine-tuning on normalized ERA5 arrays."""

        time_indices = []
        
        for year in range(self.start_year, self.end_year + 1):
            start_date = datetime(year, 1, 1, 0)
            end_date = datetime(year, 12, 31, 18)
            current_date = start_date
            
            while current_date <= end_date:
                past_time = current_date - timedelta(hours=6)
                
                
                all_files_exist = True
                
                if not os.path.exists(self._get_file_path(past_time)) or\
                   not os.path.exists(self._get_file_path(current_date)):
                    all_files_exist = False

                
                if all_files_exist:
                    for i in range(self.step):
                        future_time = current_date + timedelta(hours=6 * (i + 1))
                        if not os.path.exists(self._get_file_path(future_time)):
                            all_files_exist = False
                            break  
                
                if all_files_exist:
                    time_indices.append(current_date)
                
                current_date += timedelta(hours=6)

        return sorted(time_indices)  
        


    def _get_file_path(self, dt):
        """Dataset utilities for multi-step fine-tuning on normalized ERA5 arrays."""
        year_dir = os.path.join(self.root_dir, str(dt.year))
        date_dir = os.path.join(year_dir, dt.strftime('%Y%m%d'))

        filename = f'ERA5_Global_LM_{dt.strftime("%Y%m%d%H")}.npy' 

        return os.path.join(date_dir, filename)


    def _load_data_for_time(self, dt):
        """Dataset utilities for multi-step fine-tuning on normalized ERA5 arrays."""

        try:
            
            data_file = self._get_file_path(dt)
            data = np.load(data_file) # (84, 721, 1440)


            return torch.from_numpy(data)
        except FileNotFoundError:
            print(f"Data file not found for time {dt.strftime('%Y%m%d%H')}")
            
            
            
            return None 
        except KeyError:
            print(f"Array named 'data' was not found for time {dt.strftime('%Y%m%d%H')}")
            # raise
            return None
        except Exception as e:
            print(f"Error loading data for time {dt.strftime('%Y%m%d%H')}: {e}")
            # raise
            return None

    def __len__(self):
        return len(self.time_indices)

    def __getitem__(self, idx):
        """Dataset utilities for multi-step fine-tuning on normalized ERA5 arrays."""
        current_time = self.time_indices[idx]

        
        past_time = current_time - timedelta(hours=6)  
        past_data = self._load_data_for_time(past_time)

        current_data = self._load_data_for_time(current_time)

        input_data = torch.stack([past_data, current_data], dim=0) 

        
        labels = []
        for i in range(self.step):
            future_time = current_time + timedelta(hours=6*(i+1))
            future_data = self._load_data_for_time(future_time)
            labels.append(future_data)


        return input_data, labels

if __name__ == '__main__':
    dataset = BaselineDataset(start_year=2017, end_year=2018, step=6)
    
    if len(dataset) > 0:
        sample = dataset[0]
        if sample is not None:
            input_data, labels = sample
            print(f"Dataset length: {len(dataset)}")
            print("First sample input shape:", input_data.shape) 
            print("First sample input dtype:", input_data.dtype) 
            print("First sample output shape:", labels[0].shape) 
            print("First sample output dtype:", labels[0].dtype) 
            print("First sample slice:", input_data[0, 40, 40, 40]) # -1.0838
            print("First sample slice:", input_data.to(dtype=torch.float16)[0, 40, 30, 20]) # -1.0838
        else:
            print("Failed to load the first sample.")
    else:
        print("Dataset is empty.")
    
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=3, shuffle=True)
    
    
    for batch in dataloader:
        if batch is not None:
            batch_input, batch_output = batch
            print("\nFirst batch loaded with DataLoader:")
            print("Batch input shape:", batch_input.shape)  
            print("Batch output shape:", batch_output[0].shape)  
            print("Number of labels:", len(batch_output))
            break
        else:
            print("Failed to load batch data.")
