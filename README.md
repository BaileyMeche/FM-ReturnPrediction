# FM-ReturnPrediction

This repository contains a replication of **Lewellen (2015), "The Cross-Section of Expected Stock Returns"**, published in *Critical Finance Review*. The study explores how investors can use firm characteristics to generate real-time forecasts of a stock’s expected returns using **Fama-MacBeth regressions**.

## Overview
The project aims to replicate **Table 1, Table 2, and Figure 1** from the original paper using **CRSP and Compustat** data. It examines the predictive power of expected returns computed using multiple firm characteristics.

## Data Sources
- **CRSP** (Center for Research in Security Prices)
- **Compustat** (Standard & Poor’s financial database)


## Quick Start

To quickest way to run code in this repo is to use the following steps. First, you must have the `conda` package manager installed (e.g., via Anaconda).

Thenm open a terminal and navigate to the root directory of the project and create a 
conda environment using the following command:
```
conda create -n env_name_here python=3.12
conda activate env_name_here
```
and then install the dependencies with pip
```
pip install -r requirements.txt
```
Finally, you can then run 
```
doit
```
And that's it!


### General Directory Structure

 - Folders that start with `_` are automatically generated. The entire folder should be able to be deleted, because the code can be run again, which would again generate all of the contents. 

 - Anything in the `_data` folder (or your own RAW_DATA_DIR) or in the `_output` folder should be able to be recreated by running the code and can safely be deleted.

 - The `assets` folder is used for things like hand-drawn figures or other pictures that were not generated from code. These things cannot be easily recreated if they are deleted.

 - `_output` contains the .py generated from jupyter notebooks, and the jupyter notebooks with outputs, both in .md and in .html
 
 - `/src` contains the actual code. All notebooks in this folder will be stored cleaned from outputs (after running doit). That is in order to avoid unecessary commits from changes from simply opening or running the notebook.

 - The `data_manual` (DATA_MANUAL_DIR) is for data that cannot be easily recreated. 

 - `doit` Python module is used as the task runner. It works like `make` and the associated `Makefile`s. To rerun the code, install `doit` (https://pydoit.org/) and execute the command `doit` from the `src` directory. Note that doit is very flexible and can be used to run code commands from the command prompt, thus making it suitable for projects that use scripts written in multiple different programming languages.

 - `.env` file is the container for absolute paths that are private to each collaborator in the project. You can also use it for private credentials, if needed. It should not be tracked in Git.