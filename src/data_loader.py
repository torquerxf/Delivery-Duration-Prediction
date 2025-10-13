import pandas as pd
import zipfile
import os

def data_loader():
    """
    Extracts 'historical_data.csv' from 'data/datasets.zip' (inside datasets/ folder) and returns it as DataFrame.
    Handles exceptions for missing files or extraction errors.
    """
    zip_path = os.path.join('data', 'datasets.zip')
    csv_filename = 'datasets/historical_data.csv'
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            if csv_filename not in z.namelist():
                raise FileNotFoundError(f"'{csv_filename}' not found in the zip archive.")
            with z.open(csv_filename) as f:
                df = pd.read_csv(f)
        return df
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except zipfile.BadZipFile:
        print("Error: The zip file is corrupted or not a zip file.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return None