from __future__ import annotations
import os, json, uuid, datetime
from pathlib import Path
from typing import Any, Dict, Tuple, Optional
import numpy as np




class Utility:
    @staticmethod
    def to_bundle(distributions, domain_phis):
        '''
        Convert a list of distributions into 512-dimensional SSP bundles.
        Parameters:
            - bundles: List of vectors representing a distribution over a domain space
        Returns:
            - b: List of vectors representing a set of 512-dimensional SSP version of the distribution supplied. 
        '''
        bundles = []
        for i in range(len(distributions)):
            bundle = np.einsum('n,nd->d', distributions[i].squeeze(), domain_phis)
            bundles.append(bundle)
        return bundles
    
    @staticmethod
    def to_scale(salience, normalized_distributions):
        '''
        Scale normalized vectors by salience values.
        Parameters:
            - scales: list of floating point numbers 
            - vectors: list of vectors to be scaled 
        Returns:
            - scaled: a list of vectors scaled by a numeric scalar value
        '''
        assert len(salience) == len(normalized_distributions), "The `salience` argument has length: {len(salience)}, and `normalized_distributions` has length: {len(normalized_distributions)}. Make sure both arguments are the same length."
        scaled = []
        for i in range(len(salience)):
            scaled.append(salience[i] * normalized_distributions[i])
        return scaled

    @staticmethod
    def to_normalize(distributions):
        '''
        Normalize each distribution to unit peak (max = 1).

        Parameters: 
            - distributions (list[nd.narray]): List of distributions to be normalized.
        Returns:
            - nd.narray: List of normalized distributions with peaks at 1 
        '''
        normalized = distributions / np.max(distributions, axis=1, keepdims=True)
        return normalized
    
    @staticmethod
    def to_bimodal(distributions):
        '''
        Combine consecutive pairs of unimodal distributions into bimodal distributions.
        
        To create N beta distributions, the length of distributions should be 2N. 
        Bimodal distributions are created by combining consequetive pairs of beta distributions
        biomodal = distributions[0] + distribution[1]
        '''
        assert len(distributions) % 2 == 0, "The list `distributions` should have an even length."
        biomodals = []
        for index in range(0, len(distributions), 2):
            biomodals.append(distributions[index] + distributions[index + 1])
        return biomodals
    

class ActionIterator:
    '''
    The ActionIterator class implements a simple cyclic action scheduler and 
    feature bundler for spatial semantic pointers (SSPs).

    It is designed for scenarios where multiple actions are represented as high-dimensional vectors (SSPs), 
    and only one action is emphasized at a time while others are assigned lower salience. 
    
    The class scales SSPs by action salience and bundles them into continuous representations 
    for downstream processing (e.g., input to a basal ganglia model).

    This has been copied and modified TODO
    '''
    def __init__(self, n_actions, normalized_vectors, domain_phis):
        '''
        Parameters:
            - n_actions (int): Number of discrete actions
            - normalized_vectors (list): List of normalized SSP vectors, each of shape (512,)
            - domain_phis (nd.narray, nd.ndarray): Basis matrix used for bundling SSPs back into 512-dimensional space. 
                                                   Shape: (N, 512) where N = size of domain
        '''
        self.n_actions = n_actions
        self.actions = np.ones(n_actions) * 0.1      # initial saliences
        self.vectors = normalized_vectors            # list of 512-dim SSPs
        self.domain_phis = domain_phis               # shape (N,512) for bundling

    def step(self, t):
        '''
        Compute the bundled action vectors at time step t.
        Paramters:
            - t (float): Current time step. Used to cyclically activate actions.
        Returns:
            - np.ndarray: Concatenated vector of shape (n_actions * 512,).
                          Represents the bundled salience-weighted action SSPs for the current time step.
        '''
        # rotate which action is "on"
        idx = int(t % self.n_actions)
        self.actions[:] = 0.5
        self.actions[idx] = 0.9
        # scale each SSP by its salience
        scaled = [self.actions[i] * self.vectors[i]
                  for i in range(self.n_actions)]
        # bundle each scaled SSP back into a 512-d vector via einsum
        bundles = [
            np.einsum('n,nd->d', scaled[i].squeeze(), self.domain_phis)
            for i in range(self.n_actions)
        ]
        return np.concatenate(bundles)              # shape (n_actions * 512,)
    


class DataStorage: 
    """
    Store/load BG runs that look like:
        {
          'bundle_ins': [np.ndarray, ...],
          'd1_input': np.ndarray, 
          'd1_neuron': np.ndarray,
          'd2_input': np.ndarray, 
          'd2_neuron': np.ndarray,
          'bundle_out': np.ndarray,
          'custom_probes': { ... nested dicts/lists/ndarrays ... }
        }

    Backends:
      - NPZ: single compressed file containing arrays + JSON schema.
      - JSON: directory with manifest.json + .npy blobs for arrays.

    Stores BG probe dumps in:
      ./bg_data/npz/<run_id>.npz
      ./bg_data/json/<run_id>/manifest.json  (+ arrays/*.npy)
    """

    def __init__(self, base_dir: Optional[Path | str] = None):
        self.base_dir = Path(base_dir) if base_dir else Path.cwd() / "bg_data"
        self.json_dir = self.base_dir / "json"
        self.npz_dir  = self.base_dir / "npz"
        # ensure directory structure exists
        self.json_dir.mkdir(parents=True, exist_ok=True)
        self.npz_dir.mkdir(parents=True, exist_ok=True)

    def _now_iso(self):
        return datetime.datetime.now().replace(microsecond=0).isoformat() + "Z"
    
    def _is_ndarray(self, x): 
        return isinstance(x, np.ndarray)
    
    def _to_f32(self, a: np.ndarray) -> np.ndarray:
        a = np.asarray(a)
        if np.iscomplexobj(a):
            a = np.real_if_close(a)
        return a.astype(np.float32, copy=False)
    
    # ---------- public API ----------
    def save_npz(self, data: Dict[str, Any], meta: Dict[str, Any]=None, run_id: str|None=None, version: float = 1.0) -> Path:
        """
        Save to ./bg_data/npz/<run_id>.npz (compressed).
        Returns the written path.
        """
        run_id = run_id or self._default_run_id()
        path = self.npz_dir / f"{run_id}.npz"

        flat_arrays, schema = self._flatten_for_npz(data)
        bundle = {k: self._to_f32(v) for k, v in flat_arrays.items()}

        meta = dict(meta or {})
        meta.setdefault("created_utc", self._now_iso())
        meta.setdefault("format", "npz")
        meta.setdefault("version", f"{version}")
        meta.setdefault("run_id", run_id)

        bundle["__schema__"] = np.frombuffer(json.dumps(schema).encode("utf-8"), dtype=np.uint8)
        bundle["__meta__"]   = np.frombuffer(json.dumps(meta).encode("utf-8"),   dtype=np.uint8)

        np.savez_compressed(path, **bundle)
        return path
    
    def load_npz(self, run_id_or_path: str | Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Load from ./bg_data/npz/<run_id>.npz or an explicit .npz path.
        Returns (data, meta).
        """
        p = Path(run_id_or_path)
        if p.suffix != ".npz":
            p = self.npz_dir / f"{run_id_or_path}.npz"
        with np.load(p, allow_pickle=False) as z:
            schema = json.loads(z["__schema__"].tobytes().decode("utf-8"))
            meta   = json.loads(z["__meta__"].tobytes().decode("utf-8"))
            arrays = {k: z[k] for k in z.files if not k.startswith("__")}
        data = self._reconstruct_from_npz(arrays, schema)
        return data, meta
    
    def save_json(self, data: Dict[str, Any], meta: Dict[str, Any]=None, run_id: str|None=None, version: float = 1.0) -> Path:
        """
        Save to ./bg_data/json/<run_id>/manifest.json (+ arrays/*.npy).
        Returns the directory path for the run.
        """
        run_id = run_id or self._default_run_id()
        run_dir = self.json_dir / run_id
        arrays_dir = run_dir / "arrays"
        arrays_dir.mkdir(parents=True, exist_ok=True)

        manifest = self._serialize_to_manifest(data, arrays_dir)
        meta = dict(meta or {})
        meta.setdefault("created_utc", self._now_iso())
        meta.setdefault("format", "json+npy")
        meta.setdefault("version", f"{version}")
        meta.setdefault("run_id", run_id)

        with open(run_dir / "manifest.json", "w") as f:
            json.dump({"meta": meta, "data": manifest}, f, indent=2)
        return run_dir
    
    def load_json(self, run_id_or_dir: str | Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Load from ./bg_data/json/<run_id>/ or an explicit directory path.
        Returns (data, meta).
        """
        run_dir = Path(run_id_or_dir)
        if run_dir.suffix:  # a file was given; adjust up to parent
            run_dir = run_dir.parent
        if not run_dir.exists() or (run_dir / "manifest.json").is_file() is False:
            # interpret as run_id
            run_dir = self.json_dir / str(run_id_or_dir)
        with open(run_dir / "manifest.json", "r") as f:
            root = json.load(f)
        meta = root.get("meta", {})
        manifest = root["data"]
        data = self._deserialize_from_manifest(manifest, run_dir)
        return data, meta
    
    # ---------- helpers: NPZ flatten/reconstruct ----------
    def _flatten_for_npz(self, obj: Any, prefix: str="") -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        if self._is_ndarray(obj):
            key = prefix.rstrip("/")
            return {key: obj}, {"__type__": "ndarray", "key": key}
        if isinstance(obj, (list, tuple)):
            arrays, items_schema = {}, []
            for i, v in enumerate(obj):
                a, s = self._flatten_for_npz(v, f"{prefix}{i}/")
                arrays.update(a)
                items_schema.append(s)
            return arrays, {"__type__": "list", "items": items_schema}
        if isinstance(obj, dict):
            arrays, items_schema = {}, {}
            for k, v in obj.items():
                a, s = self._flatten_for_npz(v, f"{prefix}{k}/")
                arrays.update(a)
                items_schema[str(k)] = s
            return arrays, {"__type__": "dict", "items": items_schema}
        return {}, {"__type__": "scalar", "value": obj}

    def _reconstruct_from_npz(self, arrays: Dict[str, np.ndarray], schema: Dict[str, Any]) -> Any:
        t = schema["__type__"]
        if t == "ndarray":
            return arrays[schema["key"]]
        if t == "scalar":
            return schema.get("value", None)
        if t == "list":
            return [self._reconstruct_from_npz(arrays, s) for s in schema["items"]]
        if t == "dict":
            return {k: self._reconstruct_from_npz(arrays, s) for k, s in schema["items"].items()}
        raise ValueError(f"Unknown schema type: {t}")

    # ---------- helpers: JSON manifest (.npy blobs) ----------
    def _serialize_to_manifest(self, obj: Any, arrays_dir: Path) -> Dict[str, Any]:
        if self._is_ndarray(obj):
            fname = f"{uuid.uuid4().hex}.npy"
            np.save(arrays_dir / fname, self._to_f32(obj))
            return {"__type__": "npy", "path": f"arrays/{fname}"}
        if isinstance(obj, (list, tuple)):
            return {"__type__": "list", "items": [self._serialize_to_manifest(v, arrays_dir) for v in obj]}
        if isinstance(obj, dict):
            return {"__type__": "dict",
                    "items": {str(k): self._serialize_to_manifest(v, arrays_dir) for k, v in obj.items()}}
        return {"__type__": "scalar", "value": obj}

    def _deserialize_from_manifest(self, node: Dict[str, Any], run_dir: Path) -> Any:
        t = node.get("__type__")
        if t == "npy":
            return np.load(run_dir / node["path"])  # path is relative to run_dir
        if t == "scalar":
            return node.get("value", None)
        if t == "list":
            return [self._deserialize_from_manifest(v, run_dir) for v in node["items"]]
        if t == "dict":
            return {k: self._deserialize_from_manifest(v, run_dir) for k, v in node["items"].items()}
        raise ValueError(f"Unknown manifest type: {t}")

    # ---------- misc ----------
    def _default_run_id(self) -> str:
        return f"run-{uuid.uuid4().hex[:8]}"
    
    


