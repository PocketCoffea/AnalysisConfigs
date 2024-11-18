# MC Truth corrections for PNet pT regression

Repository to compute MC Truth corrections for PNet regressed pT jets, structured as an analysis configurations for PocketCoffea.

## Workflow

To run this over the full dataset for a particular ERA in each $\eta$ and $p_T$ bin, you can use the following command:

```bash
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

```bash
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
