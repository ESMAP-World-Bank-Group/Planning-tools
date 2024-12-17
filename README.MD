# Representative Days Project

## Disclamer

This is a recent project, with limited testing. Please report any issues you encounter.

## Overview

This project is part of the World Bank's initiative to analyze and optimize representative days for various energy models, including EPM.

The project enables users to:
- Download and parse renewables data from Renewables Ninja API.
- Include additional data sources for energy demand or hydrogeneration.
- Determine representative years for a multi-year time series.
- Calculate representative days for a given year.
- Export pHours, pDemandProfile, pVREGenProfile for the representative days.

It is based on previously developed GAMS code for the Poncelet algorithm. The objective has been to automate the process and make it more user-friendly.

## Project Structure
- `representative_days.ipynb`: Main notebook for the project. Copy and paste for your own project.
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

