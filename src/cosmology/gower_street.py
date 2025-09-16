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
        params = self.get_params_from_sim_id(serial_id, extra_params)
        return *build_cosmology(params), params

    def get_params_from_sim_id(self, serial_id, extra_params):
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
        return params
    
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
    

class GowerStPrior:
    """
    Histogram-mixture prior over parameters:
    - Builds per-parameter histogram ICDFs for multiple bin counts.
    - Samples by mixing across different binnings to smooth artifacts.
    """

    def __init__(self, bank: dict, model_params: list[str], series_true: dict[str, np.ndarray], nbins_list):
        self.bank = bank                       # model_key -> list[(edges, cdf)]
        self.model_params = model_params       # list of model_keys available
        self.series_true = series_true         # model_key -> np.ndarray of true values
        self.nbins_list = np.array(nbins_list)

    @staticmethod
    def _build_hist_icdf(s: np.ndarray, bins: int):
        s = np.asarray(s, dtype=float)
        s = s[np.isfinite(s)]
        if s.size == 0:
            return None
        pdf, edges = np.histogram(s, bins=bins, density=True)
        widths = np.diff(edges)
        pmf = pdf * widths
        pmf = pmf / pmf.sum() if pmf.sum() > 0 else np.ones_like(pmf) / pmf.size
        cdf = np.cumsum(pmf)
        return edges, cdf

    @staticmethod
    def _sample_from_hist(edges: np.ndarray, cdf: np.ndarray, n: int, rng: np.random.Generator):
        u = rng.random(n)
        idx = np.searchsorted(cdf, u, side="right")
        idx = np.clip(idx, 0, len(edges) - 2)
        left = edges[idx]
        right = edges[idx + 1]
        return left + (right - left) * rng.random(n)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, param_map: dict, nbins_list=np.arange(10, 30, 1)):
        # Build bank: model_key -> list of (edges, cdf) for each nbins
        bank = {}
        series_true = {}
        for csv_key, model_key in param_map.items():
            if csv_key not in df.columns:
                continue
            s = pd.to_numeric(df[csv_key], errors="coerce").dropna().values
            if s.size == 0:
                continue
            entries = []
            for b in nbins_list:
                icdf = cls._build_hist_icdf(s, int(b))
                if icdf is not None:
                    entries.append(icdf)
            if entries:
                bank[model_key] = entries
                series_true[model_key] = s
        model_params = list(bank.keys())
        return cls(bank, model_params, series_true, nbins_list)

    @classmethod
    def from_csv(cls, csv_path: str, drop_first: int | None = None, nbins_list=np.arange(10, 30, 1)):
        loader = GowerStCosmologies(csv_path)
        df = loader.df.copy()
        if drop_first is not None and drop_first > 0:
            df = df.iloc[drop_first:].reset_index(drop=True)
        return cls.from_dataframe(df, loader.PARAM_MAP, nbins_list=nbins_list)
    
    def _map_to_alias_keys(self, d: dict) -> dict:
        alias_by_csv = GowerStCosmologies.PARAM_MAP
        csv_by_alias = {v: k for k, v in alias_by_csv.items()}
        out = {}
        for k, v in d.items():
            if k in alias_by_csv:
                out[alias_by_csv[k]] = v         # CSV key -> alias
            elif k in csv_by_alias:
                out[k] = v                        # already alias
            else:
                out[k] = v                        # unrelated key (pass-through)
        return out

    def sample(self, n: int, rng: np.random.Generator | None = None, weights=None) -> pd.DataFrame:
        rng = rng or np.random.default_rng()
        out = {}
        for p, entries in self.bank.items():
            K = len(entries)
            w = np.asarray(weights) if weights is not None else np.ones(K) / K
            w = w / w.sum()
            comp = rng.choice(K, size=n, p=w)
            vals = np.empty(n, dtype=float)
            for k in range(K):
                mask = (comp == k)
                m = int(mask.sum())
                if m == 0:
                    continue
                edges, cdf = entries[k]
                vals[mask] = self._sample_from_hist(edges, cdf, m, rng)
            out[p] = vals
        return pd.DataFrame(out)
    def draw_param_dict_sample(self, rng: np.random.Generator | None = None, weights=None) -> dict[str, float]:
        """
        Draw a single parameter realisation as a plain dict of floats.
        """
        df = self.sample(1, rng=rng, weights=weights)
        row = df.iloc[0]
        return self._map_to_alias_keys({k: float(row[k]) for k in df.columns})