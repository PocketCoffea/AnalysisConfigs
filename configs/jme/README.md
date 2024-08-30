# PNet Regression MC Truth Correction
Repository to compute MC Truth corrections for PNet regressed pT jets, structured as an analysis configurations for PocketCoffea.


## Setup

The first step is installing the main `PocketCoffea` package in your python environment.

Please have a look at the [Installation guide](https://pocketcoffea.readthedocs.io/en/latest/installation.html).

The `configs` package has been created to separate the core of the framework from all the necessary configuration files
and customization code needed the different analyses. The configuration is structured as a python package to make easier
the import of customization code into the framework configuration and also to make the sharing of analysis code easier.

Once you have a `PocketCoffea` local installation, you can install the `configs` package with:

```python
micromamba activate pocket_coffea
pip install -e .
```

This will install the `configs` package in editable mode.


## Workflow

To run this over the full dataset for a particular ERA in each $\eta$ and $p_T$ bin, you can use the following command:
```
python exec.py --full -pnet --dir <dir_name> -y <year>
```

Year can be set to:
- 2022_preEE
- 2022_postEE
- 2023_preBPix
- 2023_postBPix

This will save the results in the `dir_name` directory inside the
`output_all.coffea` file. This file contains 2D histograms for each $\eta$ bin in which the x-axis is the jet $p_T$ response and the y-axis is the jet $p_T$.


After running the full dataset, in order to compute the MC Truth corrections, you can use the following command:
```
python response.py --full -d <dir_name>
```
This will:
- Compute the median of the response in each bin in $\eta$ as a function of $p_T$.
- Get the inverse of the median.
- Fit the inverse of the median with a 6th order polynomial.
- Save the results in the configuration file.

It will also:
- Plot the histograms of the response in each bin in $\eta$ and $p_T$ bin.
- Plot the median of the response in each bin in $\eta$ as a function of $p_T$.
- Plot the inverse of the median in each bin in $\eta$ as a function of $p_T$.
- Plot the resolution of the response in each bin in $\eta$ as a function of $p_T$ using 3 different definitions.
