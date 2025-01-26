# IRep++ implementation
This repository contains implementation and experiments with IRep++ algorithm. Project made for Machine Learning class.

Authors:
- Jakub Bąba
- Adrian Murawski

# Usage
## Prerequisites
- installed python3
- installed poetry
(if you do not have poetry installed - simply use command `pip install poetry`)

## Create virtual environment
Use this commands to create venv and install all necessary dependencies:
```bash
cd src
poetry install && poetry shell
```

## Run
There are several scripts that can be run while in poetry shell:
```bash
# creates files in data/processed/
python3 prepare_data.py
# runs experiments and writes results in results/accuracy/
python3 experiments.py
# creates plots in plots/ based on the results
python3 plot_results.py
```
