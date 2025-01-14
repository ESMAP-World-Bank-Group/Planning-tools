# Representative Days Project

## Disclamer

This is a recent project, with limited testing. Please report any issues you encounter.

## Overview

This project is part of the World Bank's initiative to analyze data input or outcome for various energy models, including EPM.

### Representative Days

This project enables users to:
- Download and parse renewables data from Renewables Ninja API.
- Include additional data sources for energy demand or hydrogeneration.
- Determine representative years for a multi-year time series.
- Calculate representative days for a given year.
- Export pHours, pDemandProfile, pVREGenProfile for the representative days.

It is based on previously developed GAMS code for the Poncelet algorithm. The objective has been to automate the process and make it more user-friendly.

The code will automatically get the min production for PV, the min production for Wind, and the max load days for each season, called the special days.
It will automatically removes the special days from the input file for the Poncelet algorithm and then runs the Poncelet algorithm to generate the representative days.
The user can decide how many representative days to generate per season.
`launch_optim_repr_days(path_data_file, folder_process_data, nbr_days=2)`

Finally, the code will merge the sepcial days with the representative day from the Poncelet algorithm and output the final representative days.

## Project Structure
- `representative_days.ipynb`:  Notebook for representative_days. Copy and paste for your own project.
- `representative_years_hydro.ipynb`: Notebook to capture representative years within full time period.
- `analysis_load.ipynb`: Notebook to analyse load curve and generate smoothed load curve to use in EPM.
- `utils.py`: Utility functions for the project.
- `data/`: Contains all the datasets used in the project (not gt-tracked).
- `parse_data/`: Includes scripts for parsing and cleaning the data (not git-tracked).
- `data_test/`: Contains test datasets for the project.
- `docs/`: Documentation and additional resources.

## Installation
To get started with the project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/RepresentativeDays.git
cd RepresentativeDays
pip install -r requirements.txt
```

If you prefer to install package manually, be aware that the GAMS API is installed with the following command:
```bash
pip install gamsapi[transfer]==xx.y.z
```
where `xx.y.z` is the version of the GAMS you use on your computer.

### Create notebook kernel from conda env
You need to create a kernel if you want to run the notebook. The idea is to first create a conda environment and then create a kernel from that environment.
After creating the conda environment, you can create a kernel for the environment. First, install the ipykernel package:

```bash
conda install ipykernel
```
Once the ipykernel package is installed, you can create a kernel for the environment:

```bash
python -m ipykernel install --user --name=esmap_env
```
