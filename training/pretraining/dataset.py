import os
import numpy as np
import torch
from torch.utils.data import Dataset
from datetime import datetime, timedelta
from config import data_dir
class BaselineDataset(Dataset):
    def __init__(self, root_dir=data_dir, start_year=1979, end_year=2016):
        """Create a pretraining dataset over normalized ERA5 NumPy files."""
        self.root_dir = root_dir
        self.start_year = start_year
        self.end_year = end_year

        
        self.time_indices = self._generate_time_indices()
        
    def _generate_time_indices(self):
        """Find timestamps with all required input and target files."""
        time_indices = []
        hours = [0, 6, 12, 18]  
        
        for year in range(self.start_year, self.end_year + 1):
            
            if year == self.start_year:
                start_date = datetime(year, 1, 1, 0)  
                
                first_day = start_date.date()
                current_date = start_date
                while current_date.date() == first_day:
                    if current_date.hour in [12, 18]:  
                        past_time = current_date - timedelta(hours=12)
                        current_exists = os.path.exists(self._get_file_path(current_date - timedelta(hours=6)))
                        past_exists = os.path.exists(self._get_file_path(past_time))
                        target_exists = os.path.exists(self._get_file_path(current_date))
                        if past_exists and current_exists and target_exists:
                            time_indices.append(current_date)
                    current_date += timedelta(hours=6)
                start_date = datetime(year, 1, 2)  
            else:
                start_date = datetime(year, 1, 1)
            

            end_date = datetime(year, 12, 31, 18)

            
            current_date = start_date
            while current_date <= end_date:
                if current_date.hour in hours:
                    
                    if current_date.hour == 6:  
                        past_time = current_date - timedelta(hours=12)  
                        if past_time.year < self.start_year:  
                            current_date += timedelta(hours=6)
                            continue
                        current_exists = os.path.exists(self._get_file_path(current_date - timedelta(hours=6)))
                    elif current_date.hour == 0:  
                        past_time = current_date - timedelta(hours=12)  
                        if past_time.year < self.start_year:  
                            current_date += timedelta(hours=6)
                            continue
                        current_exists = os.path.exists(self._get_file_path(current_date - timedelta(hours=6)))
                    else:  
                        past_time = current_date - timedelta(hours=12)
                        current_exists = os.path.exists(self._get_file_path(current_date - timedelta(hours=6)))
                    
                    
                    past_exists = os.path.exists(self._get_file_path(past_time))
                    target_exists = os.path.exists(self._get_file_path(current_date))
                    
                    if past_exists and current_exists and target_exists:
                        time_indices.append(current_date)
                
                current_date += timedelta(hours=6)
        
        return sorted(time_indices)  

    def _get_file_path(self, dt):
        """Return the normalized ERA5 NumPy file path for a timestamp."""
        year_dir = os.path.join(self.root_dir, str(dt.year))
        date_dir = os.path.join(year_dir, dt.strftime('%Y%m%d'))

        filename = f'ERA5_Global_LM_{dt.strftime("%Y%m%d%H")}.npy' 

        return os.path.join(date_dir, filename)


    def _load_data_for_time(self, dt):
        try:
            
            data_file = self._get_file_path(dt)
            data = np.load(data_file) # (69, 180, 360)
            return torch.from_numpy(data)
        except FileNotFoundError:
            print(f"Data file not found for time {dt.strftime('%Y%m%d%H')}")
            return None 
        except KeyError:
            print(f"Array named 'data' was not found for time {dt.strftime('%Y%m%d%H')}")
            return None
        except Exception as e:
            print(f"Error loading data for time {dt.strftime('%Y%m%d%H')}: {e}")
            return None

    def __len__(self):
        return len(self.time_indices)

    def __getitem__(self, idx):
        """Return two input states and the next 6-hour target state."""
        target_time = self.time_indices[idx]
        
        
        past_time = target_time - timedelta(hours=12)  
        current_time = target_time - timedelta(hours=6)  
        
        
        past_data = self._load_data_for_time(past_time)
        current_data = self._load_data_for_time(current_time)
        future_data = self._load_data_for_time(target_time)

        
        if past_data is None or current_data is None or future_data is None:
            
            print(f"Warning: failed to load complete data for index {idx} (target time {target_time}); skipping.")
            return None 

        
        
        input_data = torch.stack([past_data, current_data], dim=0) 
        
        
        output_data = future_data 
        
        
        
        return input_data, output_data

class BaselineValDataset(BaselineDataset):
    """Validation split for pretraining."""
    def __init__(self, root_dir=data_dir, start_year=2017, end_year=2017):
        """Create the validation dataset over a fixed year range."""
        
        super().__init__(root_dir=root_dir, start_year=start_year, end_year=end_year)


if __name__ == '__main__':
    dataset = BaselineDataset(start_year=1997, end_year=2016)
    
    if len(dataset) > 0:
        sample = dataset[0]
        if sample is not None:
            input_data, output_data = sample
            print(f"Dataset length: {len(dataset)}")
            print("First sample input shape:", input_data.shape) 
            print("First sample input dtype:", input_data.dtype) 
            print("First sample output shape:", output_data.shape) 
            print("First sample output dtype:", output_data.dtype) 
            print("First sample slice:", input_data[0, 40, 10, 10]) # -1.0838
            print("First sample slice:", input_data.to(dtype=torch.float16)[0, 40, 10, 10]) # -1.0838
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
            print("Batch output shape:", batch_output.shape)  
            break
        else:
            print("Failed to load batch data.")
