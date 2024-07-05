# FedHydroDSW
This repository provides an improved version of https://github.com/zxq623/FedHydro that does not require Docker.
## 0.data processing

### Download the dataset
- Download the CAMELS dataset from https://ral.ucar.edu/solutions/products/camels
- Choose one region and select seven basins from it as data-rich watersheds, and one as a data-scarce watershed.
- Place related files into the directories ./dataset/series_data/discharge_data and ./dataset/series_data/forcing_data
### Merge the dataset
- Execute ./dataset/series_data/utils/generate_data.py to generate the merged dataset
- Place the merged dataset into ./dataset/series_data/
- Create a class for loading the merged dataset in ./dataset/

## requirements
To achieve the best results, we suggest you use PyTorch==1.11.0.
## how to run


