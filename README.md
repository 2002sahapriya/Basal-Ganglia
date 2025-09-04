# Basal Ganglia
[TBD - Provide an introduction for the use case of the repo]
[ ADD PAPER LINK ]
[ DATE ]
[ COLLABORATORS ]

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

## Citation

> Note: Find function level and class level documentation in `documentation.md`

## LICENSE
- GNU