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