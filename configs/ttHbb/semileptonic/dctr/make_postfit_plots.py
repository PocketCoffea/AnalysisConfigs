import os
import sys
import re
from collections import defaultdict

from omegaconf import OmegaConf

import numpy as np
import uproot
import matplotlib.pyplot as plt
import mplhep as hep
import hist
hep.style.use("CMS")

import pocket_coffea
from pocket_coffea.utils.plot_utils import PlotManager
from pocket_coffea.parameters import defaults
import click

@click.command()
@click.option('-inp', '--input-dir', help='Directory with cofea files and parameters', type=str, default=os.getcwd(), required=False)
@click.option('-op', '--overwrite-parameters', type=str, multiple=True,
              default=None, help='YAML file with plotting parameters to overwrite default parameters', required=False)
@click.option("-o", "--outputdir", type=str, help="Output folder", required=False)
@click.option("-i", "--inputfile", type=str, help="Input file", required=False)
@click.option('-j', '--workers', type=int, default=8, help='Number of parallel workers to use for plotting', required=False)
@click.option('-oc', '--only-cat', type=str, multiple=True, help='Filter categories with string', required=False)
@click.option('-oy', '--only-year', type=str, multiple=True, help='Filter datataking years with string', required=False)
@click.option('-os', '--only-syst', type=str, multiple=True, help='Filter systematics with a list of strings', required=False)
@click.option('-e', '--exclude-hist', type=str, multiple=True, default=None, help='Exclude histograms with a list of regular expression strings', required=False)
@click.option('-oh', '--only-hist', type=str, multiple=True, default=None, help='Filter histograms with a list of regular expression strings', required=False)
@click.option('--split-systematics', is_flag=True, help='Split systematic uncertainties in the ratio plot', required=False)
@click.option('--partial-unc-band', is_flag=True, help='Plot only the partial uncertainty band corresponding to the systematics specified as the argument `only_syst`', required=False)
@click.option('-ns','--no-syst', is_flag=True, help='Do not include systematics', required=False, default=False)
@click.option('--overwrite', '--over', is_flag=True, help='Overwrite plots in output folder', required=False)
@click.option('--log', is_flag=True, help='Set y-axis scale to log', required=False, default=False)
@click.option('--density', is_flag=True, help='Set density parameter to have a normalized plot', required=False, default=False)
@click.option('-v', '--verbose', type=int, default=1, help='Verbose level for debugging. Higher the number more stuff is printed.', required=False)
@click.option('--format', type=str, default='png', help='File format of the output plots', required=False)
@click.option('--systematics-shifts', is_flag=True, help='Plot the shifts for the systematic uncertainties', required=False, default=False)
@click.option('--no-ratio', is_flag=True, help='Dont plot the ratio', required=False, default=False)
@click.option('--no-systematics-ratio', is_flag=True, help='Plot the ratio of the shifts for the systematic uncertainties', required=False, default=False)
@click.option('--compare', is_flag=True, help='Plot comparison of the samples, instead of data/MC', required=False, default=False)
@click.option('--index-file', type=str, help='Path of the index file to be copied recursively in the plots directory and its subdirectories', required=False, default=None)
@click.option('--no-cache', is_flag=True, help='Do not cache the histograms for faster plotting', required=False, default=False)

def make_plots(input_dir, overwrite_parameters, outputdir, inputfile,
               workers, only_cat, only_year, only_syst, exclude_hist, only_hist, split_systematics, partial_unc_band, no_syst,
               overwrite, log, density, verbose, format, systematics_shifts, no_ratio, no_systematics_ratio, compare, index_file, no_cache):
    '''Plot histograms produced by PocketCoffea processors'''

    # Using the `input_dir` argument, read the default config and coffea files (if not set with argparse):
    if inputfile==None:
        inputfile = os.path.join(input_dir, "postfit_shapes.root")
    if outputdir==None:
        outputdir = os.path.join(input_dir, "plots")

    # Overwrite plotting parameters
    if not overwrite_parameters:
        raise notImplementedError("Please provide the plotting parameters file to overwrite the default parameters.")
    else:
        parameters = defaults.get_defaults_and_compose(*list(overwrite_parameters))

    # Resolving the OmegaConf
    try:
        OmegaConf.resolve(parameters)
    except Exception as e:
        print("Error during resolution of OmegaConf parameters magic, please check your parameters files.")
        raise(e)

    style_cfg = parameters['plotting_style']

    if os.path.isfile( inputfile ): f = uproot.open(inputfile)
    else: sys.exit(f"Input file '{inputfile}' does not exist")

    if not overwrite:
        if os.path.exists(outputdir):
            raise Exception(f"The output folder '{outputdir}' already exists. Please choose another output folder or run with the option `--overwrite`.")

    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    folders = [k for k in f.keys() if '/' not in k]

    categories = ["CR_ttlf_0p60", "CR_ttcc", "CR", "SR"]
    years = ["2016_PreVFP", "2016_PostVFP", "2017", "2018"]

    cat_axis = hist.axis.StrCategory(categories, name="cat")
    variations_axis = hist.axis.StrCategory(["nominal", "TotalProcsUp", "TotalProcsDown"], name="variation")
    #variations_axis = hist.axis.StrCategory(["nominal"], name="variation")

    h_dict = {}
    datasets_metadata = {
        "by_datataking_period": {},
        "by_dataset": defaultdict(dict)
    }

    for stage in ["prefit", "postfit"]:
        h_dict[stage] = {}
        for cat in categories:
            index_cat = cat_axis.index(cat)
            histname = f[f"{cat}_2018_prefit/ttlf_2018;1"].to_hist().axes[0].name
            h_dict[stage][histname] = {}
            for year in years:
                folder = f"{cat}_{year}_{stage}"
                shapes_dict = f[folder]
                h_total = shapes_dict["TotalProcs;1"].to_hist()
                unc_total = np.sqrt(h_total.variances().sum())
                n_total = h_total.values().sum()
                for process_name, th1f in shapes_dict.items():
                    if process_name in ['TotalBkg;1', 'TotalProcs;1', 'TotalSig;1', 'data_obs;1']: continue
                    sample = process_name.split(f"_{year}")[0]
                    dataset = f"{sample}_{year}"
                    datasets_metadata["by_dataset"][dataset] = {'year': year, 'sample': sample, 'isMC' : 'True'}
                    if year not in datasets_metadata["by_datataking_period"]:
                        datasets_metadata["by_datataking_period"][year] = defaultdict(set)
                    datasets_metadata["by_datataking_period"][year][sample].add(dataset)
                    if sample not in h_dict[stage][histname]:
                        h_dict[stage][histname][sample] = {}
                    new_hist = hist.Hist(cat_axis, variations_axis, f[f"{cat}_2018_prefit/ttlf_2018;1"].to_hist().axes[0], storage=hist.storage.Weight())
                    new_hist_view = new_hist.view()
                    h = th1f.to_hist()
                    index_variation = variations_axis.index("nominal")
                    new_hist_view[index_cat, index_variation, :] += h.view()

                    # We compute a normalization factor to define the up and down variations such that the total uncertainty is the same as the one from the TotalProcs histogram
                    # The up/down variations are defined as follows:
                    # up = nominal * (1 + normalization_factor)
                    # down = nominal * (1 - normalization_factor)
                    # where normalization_factor = unc_total / sqrt(n_total * n_sample)
                    n_sample = h.values().sum()
                    normalization_factor = unc_total / np.sqrt(n_total * n_sample)
                    index_variation_up = variations_axis.index("TotalProcsUp")
                    new_hist_view[index_cat, index_variation_up, :] += h.view() * (1 + normalization_factor)
                    index_variation_down = variations_axis.index("TotalProcsDown")
                    new_hist_view[index_cat, index_variation_down, :] += h.view() * (1 - normalization_factor)

                    h_dict[stage][histname][sample][dataset] = new_hist

    variables = h_dict["postfit"].keys()

    if exclude_hist:
        variables_to_exclude = [s for s in variables if any([re.search(p, s) for p in exclude_hist])]
        variables = [s for s in variables if s not in variables_to_exclude]
    if only_hist:
        variables = [s for s in variables if any([re.search(p, s) for p in only_hist])]
    for stage in ["prefit", "postfit"]:
        h_dict[stage] = { v : h_dict[stage][v] for v in variables }

    for stage in ["prefit", "postfit"]:

        plotter = PlotManager(
            variables=variables,
            hist_objs=h_dict[stage],
            datasets_metadata=datasets_metadata,
            plot_dir=os.path.join(outputdir, stage),
            style_cfg=style_cfg,
            has_mcstat=False, # We set MC stat to False because we are taking the total uncertainty from the TotalProcs histogram for prefit and postfit
            only_cat=only_cat,
            only_year=only_year,
            workers=workers,
            log=log,
            density=density,
            verbose=verbose,
            save=True,
            index_file=index_file,
            cache=not no_cache
        )

        print("Started plotting.  Please wait...")

        if compare:
            plotter.plot_comparison_all(ratio=(not no_ratio), format=format)
        else:
            if systematics_shifts:
                plotter.plot_systematic_shifts_all(
                    format=format, ratio=(not no_systematics_ratio)
                )
            else:
                plotter.plot_datamc_all(syst=(not no_syst), ratio = (not no_ratio), spliteras=False, format=format)

        print("Output plots are saved at: ", outputdir)


if __name__ == "__main__":
    make_plots()
