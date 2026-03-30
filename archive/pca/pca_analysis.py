# Imports 
from copy import deepcopy
import gc
import os
import numpy as np 
from bg_model import BasalGanglia
from contrastive import CPCA
import matplotlib.pyplot as plt
from scipy.stats import beta
from utility import DataStorage, Utility
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sspspace.sspspace.encoders import RandomSSPSpace
from typing import Any, Dict, List, Optional, Sequence, Tuple


# PCA CLASS
class PCAAnalysis: 
    def __init__(self, n_components:int):
        self.n_components = n_components
        self.pca_scaler = StandardScaler()
        self.pca_model = PCA(n_components=self.n_components)
        self.datastorage = DataStorage(base_dir='bg_data')
    
    def data_format(self, data_type: str, foreground_paths, key: str, discard: int, select: int):
        # Load the data from JSON/NPZ format using DataStorage
        if data_type != 'npz' and data_type != 'json':
            raise RuntimeError('Data for PCA Analysis can only be loaded from file format of .npz or .json created using the DataStorage class format.')
        
        foreground_data = []
        for i in range(len(foreground_paths)):
            if data_type == 'npz':
                session = self.datastorage.load_npz(foreground_paths[i])
            if data_type == 'json':
                session = self.datastorage.load_json(foreground_paths[i])
            # Add session data to foreground data
            foreground_data.append(session)
        
        l3_data = []
        l12_data = []
        r3_data = []
        r12_data = []
        # Bring foreground data to the format we want for analysis 
        for i in range(len(foreground_data)):
            # 1. retrieve session data - list of 4 trials 
            session_data = foreground_data[i][0]['session_data']
            # We know each session is a list of 4 trials 
                # 1.1 iterate through the trial and retrieve the data assosicated for 'key'
                # 1.2 discard the first X rows 
                # 1.3 seperate into action channels  
            trial1 = session_data[0][key][discard:] # Shape: 150, 3000
            trial2 = session_data[1][key][discard:]
            trial3 = session_data[2][key][discard:]
            trial4 = session_data[3][key][discard:]

            t1_a1 = trial1[:, :select] # L3g wins, shape: (150, 1500)
            t2_a1 = trial2[:, :select] # L12g wins, shape: (150, 1500)
            t3_a2 = trial3[:, -select:] # R3g wins, shape: (150, 1500)
            t4_a2 = trial4[:, -select:] # R12wins, shape: (150, 1500)

            l3_data.append(t1_a1)
            l12_data.append(t2_a1)
            r3_data.append(t3_a2)
            r12_data.append(t4_a2)
        
        # Vstack the data
        left_channel = np.vstack(l3_data + l12_data) # Shape: 3000, 1500
        right_channel = np.vstack(r3_data + r12_data) # Shape: 3000, 1500

        # Create Labels 
        left_labels = (len(l3_data) * l3_data[0].shape[0])* [3] + (len(l12_data) * l12_data[0].shape[0])* [12]
        right_labels =  (len(r3_data) * r3_data[0].shape[0])* [3] + (len(r12_data) * r12_data[0].shape[0])* [12]

        return left_channel, right_channel, left_labels, right_labels
        
    def pca(self, data):
        scaler_data = self.pca_scaler.fit_transform(data)
        pca_data = self.pca_model.fit_transform(scaler_data)
        return pca_data
    
    def to_real(self, value):
        E = np.asarray(value)
        E = np.real_if_close(E).astype(np.float64)
        E[~np.isfinite(E)] = 0.0  
        return E

# Contrastive PCA CLASS
class cPCAAnalysis:
    def __init__(self, n_components: int, standardize: bool = True):
        self.n_components = n_components
        self.standardize = standardize
        self.model = CPCA(n_components=n_components, standardize=True)
        self.datastorage = DataStorage(base_dir='bg_data')
    
    def to_real(self, value):
        E = np.asarray(value)
        E = np.real_if_close(E).astype(np.float64)
        E[~np.isfinite(E)] = 0.0  
        return E
    
    def data_format(self, data_type: str, foreground_paths, key: str, discard: int, select: int):
        # Load the data from JSON/NPZ format using DataStorage
        if data_type != 'npz' and data_type != 'json':
            raise RuntimeError('Data for PCA Analysis can only be loaded from file format of .npz or .json created using the DataStorage class format.')
        
        foreground_data = []
        for i in range(len(foreground_paths)):
            if data_type == 'npz':
                session = self.datastorage.load_npz(foreground_paths[i])
            if data_type == 'json':
                session = self.datastorage.load_json(foreground_paths[i])
            # Add session data to foreground data
            foreground_data.append(session)
        
        l3_data = []
        l12_data = []
        r3_data = []
        r12_data = []
        # Bring foreground data to the format we want for analysis 
        for i in range(len(foreground_data)):
            # 1. retrieve session data - list of 4 trials 
            session_data = foreground_data[i][0]['session_data']
            # We know each session is a list of 4 trials 
                # 1.1 iterate through the trial and retrieve the data assosicated for 'key'
                # 1.2 discard the first X rows 
                # 1.3 seperate into action channels  
            trial1 = session_data[0][key][discard:] # Shape: 150, 3000
            trial2 = session_data[1][key][discard:]
            trial3 = session_data[2][key][discard:]
            trial4 = session_data[3][key][discard:]

            t1_a1 = trial1[:, :select] # L3g wins, shape: (150, 1500)
            t2_a1 = trial2[:, :select] # L12g wins, shape: (150, 1500)
            t3_a2 = trial3[:, -select:] # R3g wins, shape: (150, 1500)
            t4_a2 = trial4[:, -select:] # R12wins, shape: (150, 1500)

            l3_data.append(t1_a1)
            l12_data.append(t2_a1)
            r3_data.append(t3_a2)
            r12_data.append(t4_a2)
        
        # Vstack the data
        left_channel = np.vstack(l3_data + l12_data) # Shape: 3000, 1500
        right_channel = np.vstack(r3_data + r12_data) # Shape: 3000, 1500

        # Create Labels 
        left_labels = (len(l3_data) * l3_data[0].shape[0])* [3] + (len(l12_data) * l12_data[0].shape[0])* [12]
        right_labels =  (len(r3_data) * r3_data[0].shape[0])* [3] + (len(r12_data) * r12_data[0].shape[0])* [12]

        return left_channel, right_channel, left_labels, right_labels
    
    def cpca(self, foreground_data, background_data, labels, return_alpha: bool = True):
        return self.model.fit_transform(foreground=foreground_data, background=background_data, 
                                             active_labels=labels, return_alphas=return_alpha)
    

class Trial:
    def __init__(self, dnf_params: dict, n_actions: int, domain: np.ndarray, ssp_dim: int = 512):
        self.dnf_params = dnf_params
        self.n_actions = n_actions
        self.domain = domain.reshape(-1, 1) # Shape: (N, 1)
        self.ssp_dim = ssp_dim
        self.bg_model = None
        self.ssp_encoder = None
        self.domain_phis = None
        self.dnf_encoders = None
        self.distributions = None
        self.saliences = None
        self.trials = None
        self.bg_noise = None

    def _create_domain_phis(self, length_scale:int = 0.5, rng_seed: int = 0, domain_dim: int = 1):
        self.rng_seed = rng_seed
        self.length_scale = length_scale
        self.domain_dim = domain_dim

        self.ssp_encoder = RandomSSPSpace(
            domain_dim=self.domain_dim,
            ssp_dim=self.ssp_dim,
            rng=np.random.RandomState(self.rng_seed),
            length_scale=self.length_scale
        )
        self.domain_phis = self.ssp_encoder.encode(self.domain)
        return self.domain_phis

    def _create_dnf_encoders(self):
        low = int(min(self.domain.reshape(1, -1).squeeze())) 
        high = int(max(self.domain.reshape(1, -1).squeeze()))
        width = high - low
        self.place_cells = np.arange(low, high, width/(self.domain.shape[0]))
        self.dnf_encoders = np.asarray(self.ssp_encoder.encode(self.place_cells.reshape(-1, 1))).squeeze() # Shape: (N x 1000, ssp_dim)
        return self.dnf_encoders

    def _generate_salience_values(self,
        trial_index: int = 0,
        rng=None,
        low_min: float = 0.3,
        high_range: tuple[float, float] = (0.9, 0.99),
        eps: float = 0.05,):
        """
        Generate salience values in (0, 1] such that:
        - salience[trial_index] = high in high_range
        - all other entries are in (low_min, high) and strictly < high
        - no normalization applied

        Parameters
        ----------
        trial_index : int
            Index that must have the unique maximum.
        rng : np.random.Generator | None
            RNG for reproducibility.
        low_min : float
            Strict lower bound for non-winning saliences.
        high_range : (float, float)
            Range to sample the winning salience (low, high].
        eps : float
            Tiny margin to keep non-winners strictly below the winner.
        """
        rng = np.random.default_rng() if rng is None else rng
        n = self.n_actions * 2

        lo, hi = high_range
        if not (0.0 <= low_min < lo < hi <= 1.0):
            raise ValueError("Require 0 ≤ low_min < high_range[0] < high_range[1] ≤ 1.")

        # sample the winner's salience
        high = float(rng.uniform(lo, hi))

        # sample all others strictly below 'high' and above low_min
        # use high - eps as upper bound to guarantee strict inequality
        salience_list = rng.uniform(low_min, high - eps, size=n).astype(np.float32)

        # set the winner
        salience_list[trial_index] = high
        return salience_list

    
    def _create_bimodal_distributions(self, scale: int = 15, a_left=(2, 5), b_left=(5, 2), a_right=(2, 5), b_right=(5, 2)):
        """
        Build four one-dimensional PDFs over self.domain:
          Left:  l3 (a_left[0], b_left[0]), l12 (a_left[1], b_left[1])
          Right: r3 (a_right[0], b_right[0]), r12 (a_right[1], b_right[1])
        Note: scipy.stats.beta.pdf(x, a, b) expects x in [0,1]; if your domain is in grams
        (e.g., 0..15), ensure you've pre-normalized your domain to [0,1] or map accordingly.
        """
        # left channel
        l3 = beta.pdf(self.domain, a_left[0],  b_left[0], scale=scale)
        l12 = beta.pdf(self.domain, a_left[1],  b_left[1], scale=scale)
        # right channel
        r3 = beta.pdf(self.domain, a_right[0], b_right[0], scale=scale)
        r12 = beta.pdf(self.domain, a_right[1], b_right[1], scale=scale)
        self.distributions = [l3, l12, r3, r12]
        return self.distributions
    
    def create_bundles(self):
        """
        Create 4 trials with winner indices 0,1,2,3 respectively.
        Returns:
          trials: list of 4 bundled vectors (per trial)
          saliences: list of 4 salience arrays used
        """
        if self.distributions is None:
            raise RuntimeError("Call _create_bimodal_distributions() first.")
        if self.domain_phis is None:
            raise RuntimeError("Call _create_domain_phis() first.")
        
        trials_bundles, saliences_list = [], []
        for winner_idx in range(4):
            s = self._generate_salience_values(trial_index=winner_idx)
            saliences_list.append(s)
            # scale distributions, form bimodal sum, then bundle with domain SSPs
            scaled = Utility.to_scale(salience=s, normalized_distributions=Utility.to_normalize(self.distributions))
            bimodal = Utility.to_bimodal(scaled)
            bundle  = Utility.to_bundle(bimodal, self.domain_phis)
            trials_bundles.append(bundle)

        self.trials = trials_bundles
        self.saliences = saliences_list
        return trials_bundles, saliences_list

    def run_trial(self, d1_weight = 1.0, d2_weight = 1.0, dopamine_level = 0.2, dnf_neurons:int = 1500, presentation_time : float = 1.5, duration: float = 1.5):
        # Preconditions
        if self.domain_phis is None:
            raise RuntimeError("Call _create_domain_phis() before run_trial().")
        if self.distributions is None:
            raise RuntimeError("Call _create_bimodal_distributions() before run_trial().")
        if self.dnf_encoders is None:
            raise RuntimeError("Call _create_dnf_encoders() before run_trial().")
        if self.dnf_params is None:
            raise RuntimeError("Instantiate dnf_params before run_trial().")

        # BG Model 
        print("running Bg model now")
        self.bg_model = BasalGanglia(n_actions=self.n_actions, dnf_parameters=self.dnf_params, 
                                     encoders=self.dnf_encoders, dnf_neurons=dnf_neurons, 
                                     d1_weight=d1_weight, d2_weight=d2_weight)

        names = ["trial1", "trial2", "trial3", "trial4"]
        idx2meta = [
            {"winner": 0, "side": "L", "condition": "3g"},
            {"winner": 1, "side": "L", "condition": "12g"},
            {"winner": 2, "side": "R", "condition": "3g"},
            {"winner": 3, "side": "R", "condition": "12g"},
        ]

        results = []
        for i in range(len(self.trials)):
            trial_bundles = self.trials[i]
            probe_data = self.bg_model.simulate(input_bundles=trial_bundles, dopamine_level=dopamine_level, presentation_time=presentation_time, duration=duration)

            if not isinstance(probe_data, dict) or "d1_neuron" not in probe_data:
                raise RuntimeError("simulate() must return a dict containing key 'd1_neuron' (T×N).")
            
            results.append({
                "name": names[i],
                "meta": dict(idx2meta[i]),
                "salience": self.saliences[i],
                "bundles": trial_bundles,
                **probe_data,
            })
        
        return results

    def generate_background_data(self, dnf_neurons: int = 1500, d1_weight: float= 1.0, d2_weight: float = 1.0, high: float =10.0, rms: float =1.0, seed:int =1, duration:float = 1.5):
        if self.domain_phis is None:
            raise RuntimeError("Call _create_domain_phis() before run_trial().")
        if self.dnf_encoders is None:
            raise RuntimeError("Call _create_dnf_encoders() before run_trial().")
        if self.dnf_params is None:
            raise RuntimeError("Instantiate dnf_params before run_trial().")
       
        if self.bg_model is None:
            self.bg_model = BasalGanglia(n_actions=self.n_actions, dnf_parameters=self.dnf_params, encoders=self.dnf_encoders, 
                                         dnf_neurons=dnf_neurons, d1_weight=d1_weight, d2_weight=d2_weight)
        names = ["noise_trial1", "noise_trial2", "noise_trial3", "noise_trial4"]
        idx2meta = [
            {"winner": 0, "side": "L", "condition": "3g", "type": "noise background data"},
            {"winner": 1, "side": "L", "condition": "12g", "type": "noise background data"},
            {"winner": 2, "side": "R", "condition": "3g", "type": "noise background data"},
            {"winner": 3, "side": "R", "condition": "12g", "type": "noise background data"},
        ]

        noise_results = []
        for i in range(len(self.trials)):
            background_data = self.bg_model.simulate_noise(duration=duration, high=high, rms=rms, seed=seed)

            noise_results.append({
                "name": names[i],
                "meta": dict(idx2meta[i]),
                **background_data
            })
        return noise_results
        
    def plot_trial_distributions(
        self,
        saliences_list: Optional[List[np.ndarray]] = None,
        colors: Sequence[str] = ("#f60c89fd", "#05beb4"),
        actions_left: Sequence[str] = ("Left @ 3g", "Left @ 12g"),
        actions_right: Sequence[str] = ("Right @ 3g", "Right @ 12g"),
        peaks: Sequence[float] = (3.0, 12.0),
        combined_left_color: str = "#7609f2",
        combined_right_color: str = "#52681D",
        title_fmt: str = "Trial {idx}: {side} Channel",
        ylim: Tuple[float, float] = (0.0, 1.0),
        annotate_offset: float = 0.05,
        figsize: Tuple[int, int] = (15, 20),
        save_fig: bool = False,
        plt_name: str = None
    ):
        
        """
        Generalizes your 4×2 plotting code:
          rows = 4 trials, columns = [Left, Right]
        If `saliences_list` is None, it auto-generates winner patterns [0,1,2,3].
        Assumes self.distributions = [l3, l12, r3, r12] have been created.

        Returns
        -------
        fig, axes : matplotlib Figure and Axes array (4, 2)
        saliences_list : the 4 salience arrays used (for record)
        """
        if self.distributions is None:
            raise RuntimeError("Call _create_bimodal_distributions() first.")

        # Set up or generate saliences for 4 trials: winners 0→1→2→3
        if self.saliences is None:
            rng = np.random.default_rng()
            self.saliences = [self._generate_salience_values(trial_index=i, rng=rng) for i in range(4)]

        l3, l12, r3, r12 = self.distributions
        domain_1d = self.domain.squeeze()

        fig, axes = plt.subplots(4, 2, figsize=figsize)

         # Helper to plot a single channel pane
        def _plot_channel(ax, pdfs2, labels2, sal_slice, combined_color, side_name):
            # pdfs2: tuple/list of two PDFs for the side
            # sal_slice: saliences for those two entries
            # labels2: action labels
            # Plot the two scaled components, then the combined curve
            scaled_two = Utility.to_scale(salience=sal_slice, normalized_distributions=Utility.to_normalize(pdfs2))
            combined = scaled_two[0] + scaled_two[1]
            for i, pdf in enumerate(scaled_two):
                ax.plot(domain_1d, pdf, color=colors[i], label=f"{labels2[i]}", linestyle='-.')
                ax.axvline(x=peaks[i], color=colors[i], linestyle='--')
                # annotate salience near peak
                ax.text(peaks[i], sal_slice[i] + annotate_offset,
                        f"S={sal_slice[i]:.2f}",
                        ha='center', va='bottom', fontsize='small',
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=colors[i], alpha=0.6))
            ax.plot(domain_1d, combined, color=combined_color, label="Combined")
            ax.set_xlabel('Force (g)')
            ax.set_ylabel('Probability Density')
            ax.set_ylim(*ylim)
            ax.legend(fontsize='small')

        # Iterate over 4 trials (rows)
        for row in range(4):
            s = saliences_list[row]  # [L3, L12, R3, R12]
            # Left panel
            axes[row, 0].set_title(title_fmt.format(idx=row + 1, side="Left"), fontsize='large')
            _plot_channel(axes[row, 0], (l3, l12), actions_left, s[:2], combined_left_color, "Left")

            # Right panel
            axes[row, 1].set_title(title_fmt.format(idx=row + 1, side="Right"), fontsize='large')
            _plot_channel(axes[row, 1], (r3, r12), actions_right, s[2:], combined_right_color, "Right")

        plt.tight_layout()
        if save_fig:
            folder = './figures/pca'
            # check if folder exists
            if not os.path.isdir(folder):
                os.makedirs(folder)
            # save plots 
            if plt_name is None:
                plt_name = "trial_bimodal_beta_distributions"
            plt.savefig(f'{folder}/{plt_name}')
        return fig, axes, saliences_list
    
   
class Session: 
    def __init__(self, session_name: str, n_sessions, Trial: Trial, seed_base:int = 1234):
        self.session_name = session_name
        self.n_sessions = n_sessions
        self.trial = Trial
        self.seed_base = seed_base


    def __store_trial_data(self, session_idx: int, seed: int, trials_fg: List[Dict[str, Any]], trials_bg: List[Dict[str, Any]] = None):
        """Store the trial data (foreground required, background optional) using DataStorage."""
        if not isinstance(trials_fg, list) or len(trials_fg) == 0:
            raise RuntimeError("Foreground trials data must be a non-empty list of trial dicts.")

        # --- Build payloads exactly as DataStorage can flatten (lists/dicts/ndarrays) ---
        fg_payload: Dict[str, Any] = {
            "session_index": session_idx,
            "seed": seed,
            "session_name": self.session_name,
            "session_data": trials_fg,   # <-- list of 4 dicts: {"name","meta","salience","bundles","data":{...}}
        }
        fg_metadata = {
            "kind": "foreground",
            "session_index": session_idx,
            "seed": seed,
            "session_name": self.session_name,
            "n_trials": len(trials_fg),
        }

        bg_payload = None
        bg_metadata = None
        if trials_bg is not None:
            if not isinstance(trials_bg, list) or len(trials_bg) == 0:
                raise RuntimeError("Background trials must be a list of trial dicts when provided.")
            bg_payload = {
                "session_index": session_idx,
                "seed": seed,
                "session_name": self.session_name,
                "session_data": trials_bg,  # list of 4 dicts: {"name","meta","data":{...}}
            }
            bg_metadata = {
                "kind": "background",
                "session_index": session_idx,
                "seed": seed,
                "session_name": self.session_name,
                "n_trials": len(trials_bg),
            }

        # --- Persist using DataStorage under bg_data/pca/<session_name>/ ---
        base_dir = "bg_data"
        ds = DataStorage(base_dir=base_dir)

        sid = f"{session_idx:02d}"  # zero-pad for stable lexicographic ordering

        if not os.path.isdir(f'./bg_data/npz/pca/{self.session_name}'):
            os.makedirs(f'./bg_data/npz/pca/{self.session_name}')
        if not os.path.isdir(f'./bg_data/json/pca/{self.session_name}'):
            os.makedirs(f'./bg_data/json/pca/{self.session_name}')
        
        # Foreground: one NPZ + one JSON manifest
        fg_npz_path = ds.save_npz(data=fg_payload, meta=fg_metadata, run_id=f"pca/{self.session_name}/{sid}_foreground")
        fg_json_dir = ds.save_json(data=fg_payload, meta=fg_metadata, run_id=f"pca/{self.session_name}/{sid}_foreground")

        bg_npz_path = None
        bg_json_dir = None
        if bg_payload is not None:
            bg_npz_path = ds.save_npz(data=bg_payload, meta=bg_metadata, run_id=f"pca/{self.session_name}/{sid}_background")
            bg_json_dir = ds.save_json(data=bg_payload, meta=bg_metadata, run_id=f"pca/{self.session_name}/{sid}_background")

        return {
            "npz": [str(fg_npz_path), (str(bg_npz_path) if bg_npz_path is not None else None)],
            "json": [str(fg_json_dir), (str(bg_json_dir) if bg_json_dir is not None else None)],
        }


    def run(self, duration: float, presentation_time: float) -> List[str]:
        'Run the N sessions: generate foreground and background data'
        written_paths: List[str] = []
        
        for s in range(self.n_sessions):
            # fresh copy of the template Trial
            trial = deepcopy(self.trial)

            # build encoders and distributions; reseed per session
            trial._create_domain_phis(length_scale=0.5, rng_seed=self.seed_base + s, domain_dim=1)
            trial._create_dnf_encoders()
            trial._create_bimodal_distributions()
            trial.create_bundles()  # prepares 4 bundles + saliences
            _, _, _  = trial.plot_trial_distributions(saliences_list=trial.saliences, save_fig=True, plt_name=f'pca_session_{s}')

            # foreground data: run 4 trials on one BG instance
            trials_fg = trial.run_trial(duration=duration, presentation_time=presentation_time) # list of 4 dicts with probe data

            # background data: run 4 traials on one BG instance
            trials_bg = trial.generate_background_data(duration=duration) # list of 4 dicts with probe data
            # trials_bg = None
            # save and free memory
            written_paths.append(self.__store_trial_data(
                session_idx=s,
                trials_fg=trials_fg,
                trials_bg=trials_bg,
                seed=self.seed_base + s,
            ))

            del trial, trials_fg, trials_bg
            gc.collect()

        return written_paths


