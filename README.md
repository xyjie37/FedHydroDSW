# FedHydroDSW
This repository provides an improved version of https://github.com/zxq623/FedHydro that does not require Docker.
  
  “Improving Adaptive Runoff Forecasts in Data-Scarce Watersheds Through Personalized Federated Learning” ICPR2024，oral
## Data processing

### Download the dataset
- Download the CAMELS dataset from https://ral.ucar.edu/solutions/products/camels
- Choose one region and select seven basins from it as data-rich watersheds, and one as a data-scarce watershed.
- Place related files into the directories ./dataset/series_data/discharge_data and ./dataset/series_data/forcing_data
### Merge the dataset
- Execute ./dataset/series_data/utils/generate_data.py to generate the merged dataset
- Place the merged dataset into ./dataset/series_data/
- Create a class for loading the merged dataset in ./dataset/

## Requirements
To achieve the best results, we suggest using PyTorch == 1.11.0.
## How to run
You can adjust the parameters in `utils/options.py`, and then run `main_fedhydro.py`.
## Pretrained Model
We provide two pretrained models on basin18,the code used to be test the model is under folder /utils

