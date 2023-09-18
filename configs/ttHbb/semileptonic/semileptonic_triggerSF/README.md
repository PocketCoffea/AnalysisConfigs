# Semileptonic single electron trigger SF

The current guide is a documentation of all the steps that need to be followed in order to compute the semileptonic single electron trigger SF for the $ttH(\rightarrow b\bar{b})$ analysis with Ultra Legacy (UL) datasets.

Each data-taking era needs to be processed separately by writing a config file.

## Build dataset

The first step is to build the json files containing the list of files of the UL datasets to process. In our case, we need to include the `TTToSemiLeptonic` and `TTTo2L2Nu` MC datasets and `SingleMuon` as data dataset.

A step-by-step detailed guide on the creation of the dataset file can be found at this [link](https://pocketcoffea.readthedocs.io/en/latest/examples.html).

## Include analysis-specific parameters

To compute the semileptonic trigger SF in the ttH(bb) analysis, the standard UL parameters are used, taken from `pocket_coffea.parameters.defaults`.
The only custom parameters that we need to define in order to run the analysis are:

- HLT trigger paths: defined in `params/triggers.yaml`, for each data-taking era
- Object preselection: defined in `params/object_preselection_semileptonic.yaml`

## Write config file

All the parameters specific to the analysis need to be specified in a config file that is passed as an input to the `runner.py` script.

For each data-taking era, there is a dedicated config file:

- 2016_PreVFP: `semileptonic_triggerSF_2016_PreVFP.py`
- 2016_PostVFP: `semileptonic_triggerSF_2016_PostVFP.py`
- 2017: `semileptonic_triggerSF_2017.py`
- 2018: `semileptonic_triggerSF_2018.py`

## Run the analysis

In order to run the analysis workflow and produce the output histograms, run the following command:

```
cd /path/to/AnalysisConfigs/configs/ttHbb/semileptonic/semileptonic_triggerSF
runner.py --cfg semileptonic_triggerSF/semileptonic_triggerSF_2016_PreVFP.py --full
runner.py --cfg semileptonic_triggerSF/semileptonic_triggerSF_2016_PostVFP.py --full
runner.py --cfg semileptonic_triggerSF/semileptonic_triggerSF_2017.py --full
runner.py --cfg semileptonic_triggerSF/semileptonic_triggerSF_2018.py --full
```

N.B.: the argument `--full` will process all the datasets together at once and save the output in a single output file, `output_all.coffea`. Otherwise, the datasets are processed separately and an output file is saved for each dataset.

## Accumulate output files

If the output of the [previous step](#run-the-analysis) has been produced without the argument `--full`, the output files need to be merged in a single output file `output_all.coffea`. If the output has been produced with the argument `--full` and the output file `output_all.coffea` is already existing, skip this step and continue with the [next one](#produce-datamc-plots).

Once the Coffea output files are produced, one needs to merge the files into a single file by using the script `accumulate_files.py` by running this command:

```
cd /path/to/output
accumulate_files.py -i output1.coffea output2.coffea output3.coffea -o output_all.coffea
```

## Produce data/MC plots

Once the Coffea output has been accumulated, the plots can be produced by executing the plotting script:

```
cd /path/to/output/folder
make_plots.py --cfg parameters_dump.yaml -op plotting_style.yaml -i output_all.coffea -o plots -j 8
```

The `make_plots.py` takes as arguments:

- `--cfg`: The .yaml config file dumped after running the analysis, containing all the analysis parameters
- `-op`: Additional .yaml config file to overwrite the plotting style when plotting
- `-i`: Input .coffea file with histograms
- `-j`: Number of workers used for plotting
- `--overwrite`: If the output folder is already existing, overwrite its content

## Run trigger SF script

To run the script that computes the single electron trigger SF and produces the validation plots, run:

```
cd /path/to/output/folder
trigger_efficiency.py --cfg parameters_dump.yaml -op /path/to/AnalysisConfigs/configs/ttHbb/semileptonic/semileptonic_triggerSF/params/plotting_style_efficiency_maps.yaml -i output_all.coffea -o trigger_sf -j 8 --save_plots
```

The `make_plots.py` takes as arguments:

- `--cfg`: The .yaml config file dumped after running the analysis, containing all the analysis parameters
- `-op`: Additional .yaml config file to overwrite the plotting style when plotting, with parameters specific to the trigger SF plots
- `-i`: Input .coffea file with histograms
- `-o`: Output folder to save the correction maps, the efficiency and scale factor plots
- `-j`: Number of workers used for plotting
- `--save_plots`: if not specified as argument, only the correction maps are saved without plots
- `--overwrite`: If the output folder is already existing, overwrite its content

The output plots are saved in `trigger_sf/trigger_efficiency` and `trigger_sf/trigger_scalefactor`, while the 1D and 2D scale factor maps are saved in the folder `trigger_sf/correction_maps`.
