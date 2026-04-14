# Basal Ganglia
Status: Accepted -- Submitted to Cognitive Science Society 2026] \
Date: February 2, 2026 \
Authors: Priyadarshini Saha, Madeliene Bartlett, Jeff Orchard \
[Paper Link]

## Requirements 
- Python == 3.10.18
- gymnasium==1.0.0
- matplotlib==3.10.5
- nengo==4.1.0
- numpy==1.26.4
- scipy==1.16.1
- setuptools==80.9.0
- nengo_dft == 0.0.1 (from https://github.com/tcstewar/nengo-dft/tree/main)
- sspspace == 0.1 (from https://github.com/ctn-waterloo/sspspace)

## Installation Instructions 

#### 1. Installing `nengo-dft` as a local package
- Clone the git repo:
  ```python
  git clone https://github.com/tcstewar/nengo-dft.git
  ```
- Change Director to the `nengo-dft1` repo:
  ```python
  cd nengo-dft
  ```
- Install the `nengo-dft` repository as a local package
  ```python
  pip install -e .
  ```
- Rename `nengo-dft` as `nengo_dft`

#### 2. Installing `sspspace` 
- Clone the git repo:
  ```python
  git clone https://github.com/ctn-waterloo/sspspace.git
  ```

#### 3. Running `requirements.txt`
- Make sure you are in the same folder as `requirements.txt`
- Run: `pip install -r requirements.txt`

## Model
Architecture of the Basal Ganglia model
![BG Network](figures/bg_network.png "Figure 1: Basal Ganglia Network Architecture")

## Documentation

Full documentation lives in the [`docs/`](docs/) folder.

### Core Model

| Document | Source file | Description |
|---|---|---|
| [`docs/bg_model.md`](docs/bg_model.md) | `bg_model.py`, `utility.py` | API reference for all classes and methods: `DNF`, `BasalGanglia`, `Utility`, `ActionIterator`, `DataStorage`. |
| [`docs/bg_model_demo.md`](docs/bg_model_demo.md) | `bg_model.ipynb` | Step-by-step walkthrough of the 4-action-channel demo — domain setup, SSP encoding, bundle construction, simulation, saving, and result plots. |

### Park Task Experiments

The Park Task tests BG action selection between two motor actions (Lever Pull vs. Button Press) over a continuous force domain `[0, 15]g`, using a 2×2 salience trial design.

| Document | Source file | Description |
|---|---|---|
| [`docs/park_task_unimodal.md`](docs/park_task_unimodal.md) | `park_task.ipynb` | Baseline experiment: one Beta distribution peak per action channel (unimodal). |
| [`docs/park_task.md`](docs/park_task.md) | `park_bimodal.ipynb` | Extended experiment: two peaks per channel (bimodal), testing selection under distributional ambiguity. |

## Citation

## LICENSE
- GNU
