# `bg_model.py` — API Reference

This module implements the core Basal Ganglia (BG) model used in the paper.
It contains two classes: `DNF` (a helper for Dynamic Neural Fields) and `BasalGanglia` (the full BG network).

---

## `DNF`

A static helper class for constructing Dynamic Neural Field (DNF) kernels and Nengo networks.

---

### `DNF.make_kernel(shape, exc, inh, exc_width=5, inh_width=10, epsilon=0.001)`

Constructs a lateral interaction kernel (excitatory minus inhibitory Gaussian) for a 1D or 2D DNF.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `shape` | `tuple` | Shape of the field — `(N,)` for 1D or `(N, N)` for 2D. |
| `exc` | `float` | Excitatory scaling factor (total area under excitatory Gaussian). |
| `inh` | `float` | Inhibitory scaling factor (total area under inhibitory Gaussian). |
| `exc_width` | `float` | Standard deviation (σ) of the excitatory Gaussian. Default: `5`. |
| `inh_width` | `float` | Standard deviation (σ) of the inhibitory Gaussian. Default: `10`. |
| `epsilon` | `float` | Tail cutoff threshold for truncating the kernel. Default: `0.001`. |

**Returns**

`np.ndarray` — The difference kernel `k_exc - k_inh`, ready to use as a convolutional weight matrix.

---

### `DNF.make_dnf(shape, tau, c_noise, beta, global_inh, h, exc, inh, exc_w, inh_w, dt)`

Builds a Nengo network implementing a full Dynamic Neural Field with recurrent lateral interactions and global inhibition.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `shape` | `tuple` | Dimensions of the neural field, e.g. `(100,)` or `(32, 32)`. |
| `tau` | `float` | Synaptic time constant for recurrent connections. |
| `c_noise` | `float` | Amplitude of background white noise (set to `0` to disable). |
| `beta` | `float` | Gain parameter for the sigmoid activation function. |
| `global_inh` | `float` | Global (uniform) inhibitory weight applied across all neurons. |
| `h` | `float` | Resting potential / bias added to the global inhibition node. |
| `exc` | `float` | Excitatory kernel scaling factor. |
| `inh` | `float` | Inhibitory kernel scaling factor. |
| `exc_w` | `float` | Excitatory kernel width. |
| `inh_w` | `float` | Inhibitory kernel width. |
| `dt` | `float` | Simulation timestep (used to scale the input connection). |

**Returns**

`nengo.Network` — A Nengo network with two exposed attributes:
- `net.g` — The main neural ensemble (Sigmoid neurons).
- `net.u` — The external input node (size = `prod(shape)`).

**Network topology**

```
external input (net.u) ──► g.neurons (recurrent, lateral convolution)
                                │
                          global_inh node ◄──── g.neurons (summed)
                                │
                           h (bias node) ──►
```

---

## `BasalGanglia`

A biologically inspired Basal Ganglia model implemented as a `nengo.Network`. It maps cortical SSP inputs through D1/D2 striatal DNFs and subcortical nuclei (STN, GPe, GPi) to produce disinhibited output bundles.

---

### Constructor

```python
BasalGanglia(
    n_actions,
    dnf_parameters,
    encoders,
    d1_weight=1.0,
    d2_weight=1.0,
    neuron_type=LIFRate(),
    seed=None,
    dnf_neurons=400
)
```

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `n_actions` | `int` | — | Number of discrete action channels. |
| `dnf_parameters` | `dict` | — | Dict of kwargs passed directly to `DNF.make_dnf()`. |
| `encoders` | `np.ndarray` | — | SSP place-cell encoders, shape `(domain_size, ssp_dim)`. Used to project SSP vectors onto DNF neurons. |
| `d1_weight` | `float` | `1.0` | Scaling factor for dopamine modulation in D1 pathway: `segment * (d1_weight + dopamine)`. |
| `d2_weight` | `float` | `1.0` | Scaling factor for dopamine modulation in D2 pathway: `segment * (d2_weight - dopamine)`. |
| `neuron_type` | `NeuronType` | `LIFRate()` | Nengo neuron model used for STN, GPe, GPi ensembles. |
| `seed` | `int` | `None` | Random seed for reproducibility. |
| `dnf_neurons` | `int` | `400` | Number of neurons allocated per action channel within each DNF. |

**Key attributes after construction**

| Attribute | Type | Description |
|---|---|---|
| `cortex_inputs` | `list[Node]` | One input node per action channel, each of size `ssp_dim` (512). |
| `dopamine` | `Node` | Scalar dopamine input node. |
| `concentration_layer` | `Ensemble` | Direct-mode ensemble that concatenates all action SSPs + dopamine signal. |
| `d1_dnf`, `d2_dnf` | `nengo.Network` | DNF networks for D1 and D2 striatal pathways. |
| `stn`, `gpe`, `gpi` | `list[Ensemble]` | Per-action subcortical nuclei (1000 neurons × 512D each). |
| `bg_out` | `Node` | Output node collecting GPi projections, size `n_actions × ssp_dim`. |

**Circuit connectivity**

```
cortex_inputs[i] ──┐
                   ├──► concentration_layer ──► D1 DNF ──► GPi[i] ──► bg_out
dopamine ──────────┘                       └──► D2 DNF ──► GPe[i] ──┐
                                                                      │
cortex_inputs[i] ─────────────────────────────► STN[i] ──────────────┤
                                                   │                  │
                                               GPe[i] ◄──────────────┘
                                                   │
                                               GPi[i] ──► bg_out
```

Connection signs: D1→GPi (−), D2→GPe (−), STN→GPi (+), STN→GPe (+), GPe→STN (−), GPe→GPi (−), GPi→output (−3×).

---

### `add_probe(part, index=None, what='output', field=None, name=None)`

Queue a probe to be created when `simulate()` or `simulate_noise()` is called. Probes are deferred because Nengo requires probes to be added inside an active network context.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `part` | `str` | Which component to probe. One of: `'cortex_inputs'`, `'stn'`, `'gpe'`, `'gpi'`, `'concentration_layer'`, `'dopamine'`, `'bg_out'`, `'d1_dnf'`, `'d2_dnf'`. |
| `index` | `int \| None` | For per-action lists, which channel (0 … n_actions−1). `None` probes all channels. |
| `what` | `str` | `'output'` (decoded), `'input'` (pre-decode), or `'neurons'` (raw neuron signal). |
| `field` | `str \| None` | When `what='neurons'`: `None` (firing rates), `'input'` (input current), `'spikes'`, or `'voltage'`. |
| `name` | `str \| None` | Key name for the probe in the returned `custom_probes` dict. Auto-generated if omitted. |

**Returns** `str` — The key name that will appear in `data['custom_probes']`.

---

### `simulate(input_bundles, dopamine_level=0.0, presentation_time=1.5, duration=1.0)`

Run the BG network with structured SSP inputs.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `input_bundles` | `list[np.ndarray]` | — | List of `n_actions` SSP vectors, each of shape `(512,)`. |
| `dopamine_level` | `float` | `0.0` | Scalar dopamine value presented for the full simulation. |
| `presentation_time` | `float` | `1.5` | Duration (s) for which inputs are held constant via `PresentInput`. |
| `duration` | `float` | `1.0` | Total simulation duration (s). |

**Returns** `dict` with keys:

| Key | Shape | Description |
|---|---|---|
| `bundle_ins` | `list[T×512]` | Probed input bundles per action channel. |
| `d1_input` | `T × (n_actions×dnf_neurons)` | Input current to D1 DNF neurons. |
| `d1_neuron` | `T × (n_actions×dnf_neurons)` | D1 DNF neuron activities (firing rates). |
| `d2_input` | `T × (n_actions×dnf_neurons)` | Input current to D2 DNF neurons. |
| `d2_neuron` | `T × (n_actions×dnf_neurons)` | D2 DNF neuron activities. |
| `bundle_out` | `T × (n_actions×512)` | BG output bundles (concatenated across action channels). |
| `custom_probes` | `dict` | Any probes queued via `add_probe()`. |

---

### `simulate_noise(duration=1.5, high=10.0, rms=1.0, seed=0)`

Run the BG network with band-limited white noise as cortical input and dopamine clamped to 0. Used to characterise the network's spontaneous dynamics.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `duration` | `float` | `1.5` | Total simulation duration (s). |
| `high` | `float` | `10.0` | Cutoff frequency (Hz) for the band-limited noise. |
| `rms` | `float` | `1.0` | RMS amplitude of the noise (tune relative to SSP magnitude). |
| `seed` | `int` | `0` | Base random seed; each action channel gets `seed + i`. |

**Returns** Same dict structure as `simulate()`.

---

## `Utility` (from `utility.py`)

Static helper methods for preparing distributions and SSP bundles.

| Method | Description |
|---|---|
| `Utility.to_normalize(distributions)` | Normalize each distribution to unit peak (max = 1). |
| `Utility.to_scale(salience, normalized_distributions)` | Scale each normalized distribution by its corresponding salience scalar. |
| `Utility.to_bundle(distributions, domain_phis)` | Convert a list of weighted distributions into 512D SSP bundles via `einsum`. |
| `Utility.to_bimodal(distributions)` | Combine consecutive pairs of unimodal distributions into bimodal distributions. |

## `ActionIterator` (from `utility.py`)

Cyclic action scheduler that rotates salience across action channels over time and bundles their SSPs.

```python
it = ActionIterator(n_actions, normalized_vectors, domain_phis)
bundle = it.step(t)  # shape: (n_actions * 512,)
```

## `DataStorage` (from `utility.py`)

Saves and loads BG simulation probe data. Supports two backends:
- **NPZ** (`save_npz` / `load_npz`): single compressed `.npz` file with embedded JSON schema.
- **JSON + NPY** (`save_json` / `load_json`): directory with `manifest.json` and per-array `.npy` blobs.

```python
ds = DataStorage(base_dir='bg_data')
path = ds.save_npz(result, meta=meta, run_id='my_run')
data, meta = ds.load_npz('my_run')
```
