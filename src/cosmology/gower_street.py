import csv
import numpy as np
import pandas as pd
import camb
from typing import Dict, Tuple
from .parameters import build_cosmology
from .simulators import GowerStreetSimulator

class GowerStCosmologies:
    PARAM_MAP = {
        "little_h": "h",
        "Omega_b little_h^2": "ombh2",
        "sigma_8": "s8",
        "w": "w0",
        "n_s": "ns",
        "m_nu": "mnu",
        "Omega_m": "omega_m"
    }

    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path, skiprows=1)  # Skips first line of extra header
        self.df = self._clean_dataframe(self.df)

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        # Strip and clean column names
        df.columns = df.columns.str.strip()

        # Convert all values to numeric where possible
        df = df.apply(pd.to_numeric, errors='ignore')

        return df

    def get_simulation_cosmology(self, serial_id: int, extra_params, **kwargs):
        if serial_id not in self.df["Serial Number"].values:
            raise KeyError(f"Serial ID {serial_id} not found in dataset.")
        
        row = self.df[self.df["Serial Number"] == serial_id].iloc[0]

        # Map parameters needed by build_cosmology
        params = {}
        for csv_key, model_key in self.PARAM_MAP.items():
            if csv_key in row:
                val = row[csv_key]
                if pd.notna(val):
                    params[model_key] = float(val)
        params = {**params, **extra_params}
        return *build_cosmology(params), params
    
class GowerStDatasetBuilder:

    def __init__(self, csv_path: str, dataset_path: str):
        self.cosmology_loader = GowerStCosmologies(csv_path)
        self.dataset_path = dataset_path
    
    def get_simulation_cosmology(self, serial_id: int, *args, **kwargs):
        return self.cosmology_loader.get_simulation_cosmology(serial_id, *args,**kwargs)
    
    def setup_simulator(self, serial_id: int, **simulation_kwargs):
        """
        Build a Gower Street dataset map for a given serial ID.
        """
        sim_path = f"{self.dataset_path}/sim{serial_id:05d}"
        simulator = GowerStreetSimulator(sim_path, **simulation_kwargs)
        
        return simulator