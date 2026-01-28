'''Script to extract the tt+LF background calibration and plot the shapes comparison.

The SF is exported in correctionlib json format. 
'''

import numpy as np
import awkward as ak
import hist
from itertools import product
from multiprocessing import Pool
from coffea.util import load
import os

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize
import mplhep as hep

hep.style.use(hep.style.ROOT)

from pocket_coffea.utils.plot_functions import plot_shapes_comparison

import argparse

# Set the matplotlib backend to 'Agg' to avoid displaying plots
plt.switch_backend('Agg')
dpi = 300

# Taken from https://gist.github.com/kdlong/d697ee691c696724fc656186c25f8814
def rebin_hist(h, axis_name, edges):
    if type(edges) == int:
        return h[{axis_name : hist.rebin(edges)}]

    ax = h.axes[axis_name]
    ax_idx = [a.name for a in h.axes].index(axis_name)
    if not all([np.isclose(x, ax.edges).any() for x in edges]):
        raise ValueError(f"Cannot rebin histogram due to incompatible edges for axis '{ax.name}'\n"
                            f"Edges of histogram are {ax.edges}, requested rebinning to {edges}")
        
    # If you rebin to a subset of initial range, keep the overflow and underflow
    overflow = ax.traits.overflow or (edges[-1] < ax.edges[-1] and not np.isclose(edges[-1], ax.edges[-1]))
    underflow = ax.traits.underflow or (edges[0] > ax.edges[0] and not np.isclose(edges[0], ax.edges[0]))
    flow = overflow or underflow
    new_ax = hist.axis.Variable(edges, name=ax.name, overflow=overflow, underflow=underflow)
    axes = list(h.axes)
    axes[ax_idx] = new_ax
    
    hnew = hist.Hist(*axes, name=h.name, storage=h._storage_type())

    # Offset from bin edge to avoid numeric issues
    offset = 0.5*np.min(ax.edges[1:]-ax.edges[:-1])
    edges_eval = edges+offset
    edge_idx = ax.index(edges_eval)
    # Avoid going outside the range, reduceat will add the last index anyway
    if edge_idx[-1] == ax.size+ax.traits.overflow:
        edge_idx = edge_idx[:-1]

    if underflow:
        # Only if the original axis had an underflow should you offset
        if ax.traits.underflow:
            edge_idx += 1
        edge_idx = np.insert(edge_idx, 0, 0)

    # Take is used because reduceat sums i:len(array) for the last entry, in the case
    # where the final bin isn't the same between the initial and rebinned histogram, you
    # want to drop this value. Add tolerance of 1/2 min bin width to avoid numeric issues
    hnew.values(flow=flow)[...] = np.add.reduceat(h.values(flow=flow), edge_idx, 
            axis=ax_idx).take(indices=range(new_ax.size+underflow+overflow), axis=ax_idx)
    if hnew._storage_type() == hist.storage.Weight():
        hnew.variances(flow=flow)[...] = np.add.reduceat(h.variances(flow=flow), edge_idx, 
                axis=ax_idx).take(indices=range(new_ax.size+underflow+overflow), axis=ax_idx)
    return hnew

def plot_shape(args):
    output_variables, var, sample, year, args_output = args
    print(f"Plotting {var} {sample} {year}")
    shapes = [
        (sample, cat, year, "nominal", cat),
    ]
    plot_shapes_comparison(output_variables, f"{var}", shapes, ylog=True,
                           lumi_label=f"{sample} {year}",
                           outputfile=os.path.join(args_output, f"{var}_{sample}_{year}.png"))
    
# Function to annotate bins with their values
def annotate_bins(ax, hist):
    for i in range(hist.axes[0].size):
        for j in range(hist.axes[1].size):
            value = hist.values()[i, j]
            if value >= 0:  # Only annotate bins with positive or zero values
                ax.text(hist.axes[0].centers[i], hist.axes[1].centers[j], f"{value:.2f}", 
                        ha='center', va='center', color='white', fontsize=12)

parser = argparse.ArgumentParser(description="Extract btagSF calibration and validate it with plots")
parser.add_argument("-i","--input", type=str, required=True, help="Input coffea files with shapes")
parser.add_argument("-o","--output", type=str, required=True, help="Output folder")
parser.add_argument("-v","--validate", action="store_true", help="Use this switch to plot validation shapes")
parser.add_argument("-c","--compute", action="store_true", help="Use this switch to compute the SF")
parser.add_argument("--sf-hist", type=str, help="Histogram to be used for SF computation", default="Njet_Ht")
parser.add_argument("-j", "--workers", type=int, default=8, help="Number of workers for parallel processing")
parser.add_argument("--no-plots", action="store_true", help="Do not produce plots")
args = parser.parse_args()

os.makedirs(args.output, exist_ok=True)
output = load(args.input)
output_variables = output['variables']
h = output_variables[f"{args.sf_hist}"]

variables_to_plot = [
    'jets_Ht', "nJets", "nBJets",
    "JetGood_pt_1", "JetGood_eta_1", #"JetGood_btagDeepFlavB_1",
    "JetGood_pt_2", "JetGood_eta_2", #"JetGood_btagDeepFlavB_2",
    "JetGood_pt_3", "JetGood_eta_3", #"JetGood_btagDeepFlavB_3",
    "JetGood_pt_4", "JetGood_eta_4", #"JetGood_btagDeepFlavB_4",
]

years = list(output["datasets_metadata"]['by_datataking_period'])
samples = h.keys()
samples_ttlf = ["TTToSemiLeptonic__TTToSemiLeptonic_tt+LF"]
samples_ttcc = ["TTToSemiLeptonic__TTToSemiLeptonic_tt+C"]
samples_data = [s for s in samples if s.startswith("DATA")]
variations = ["nominal", "norm_ttccUp", "norm_ttccDown"]
cat = "semilep"
cat_calib = "semilep_ttlf_calib"
cmap = "viridis"

if args.compute:
    if not args.no_plots:
        # Prepare the arguments for parallel processing
        plot_args = [
            (output_variables, var, sample, year, args.output)
            for var, sample, year in product(variables_to_plot, samples, years)
            if sample not in samples_data
        ]

        # Use multiprocessing to parallelize the plotting
        with Pool(args.workers) as pool:
            pool.map(plot_shape, plot_args)

    # Compute the SF in one go
    samples = list(h.keys())
    for year in years:
        exclude_samples = [
            "ttHTobb_ttToSemiLep",
            "TTToSemiLeptonic__TTToSemiLeptonic_tt+B",
            "TTbbSemiLeptonic__TTbbSemiLeptonic_tt+LF",
            "TTbbSemiLeptonic__TTbbSemiLeptonic_tt+C"
        ]
        samples_other = [s for s in samples if s not in samples_ttlf+samples_ttcc+samples_data+exclude_samples]
        mask_data = {'cat':cat}
        mask_mc = mask_data | {'variation':'nominal'}
        h_data = sum({k :val for s in samples_data for k, val in h[s].items() if year in k}.values())[mask_data]
        h_ttlf = sum({k :val for s in samples_ttlf for k, val in h[s].items() if year in k}.values())[mask_mc]
        h_ttcc = sum({k :val for s in samples_ttcc for k, val in h[s].items() if year in k}.values())[mask_mc]
        h_other = sum({k :val for s in samples_other for k, val in h[s].items() if year in k}.values())[mask_mc]

        # Rebin the histogram to merge the last two bins of Njet
        njet_axis = h_data.axes[0]
        njet_edges = njet_axis.edges
        jets_ht_axis = h_data.axes[1]
        jets_ht_edges = jets_ht_axis.edges

        # Define new bin edges that merge the last two bins
        new_njet_edges = [4,5,6,7,8,9,20]
        nbins_to_merge = 4
        new_ht_edges = np.concatenate((jets_ht_edges[:-nbins_to_merge], [jets_ht_edges[-1]]))
        print("Njet bins:", new_njet_edges)
        print("HT bins:", new_ht_edges)

        # Rebin the histogram
        for axis, new_edges in zip([njet_axis, jets_ht_axis], [new_njet_edges, new_ht_edges]):
            h_data = rebin_hist(h_data, axis.name, new_edges)
            h_ttlf = rebin_hist(h_ttlf, axis.name, new_edges)
            h_ttcc = rebin_hist(h_ttcc, axis.name, new_edges)
            h_other = rebin_hist(h_other, axis.name, new_edges)

        # Compute the SF and the up and down variations obtained by varying the tt+cc background
        h_ttcc_up = h_ttcc * 1.5
        h_ttcc_down = h_ttcc * 0.5
        h_num = hist.Hist(
            hist.axis.StrCategory(variations, name="variation"),
            *h_data.axes,
            storage=hist.storage.Weight(),
        )
        h_num_view = h_num.view(flow=True)
        h_num_view[0,:,:] = (h_data + -1*h_other + -1*h_ttcc).view(flow=True)
        h_num_view[1,:,:] = (h_data + -1*h_other + -1*h_ttcc_up).view(flow=True)
        h_num_view[2,:,:] = (h_data + -1*h_other + -1*h_ttcc_down).view(flow=True)
        h_denom = h_ttlf
        stack_ratio = []
        stack_ratio_err = []
        for variation in variations:
            w_num, x, y = h_num[{'variation':variation}].to_numpy()
            var_num = h_num[{'variation':variation}].variances()
            w_denom, x, y = h_denom.to_numpy()
            var_denom = h_denom.variances()

            ratio= np.where( (w_denom>0)&(w_num>0),
                                w_num/w_denom,
                                1.)
            ratio = np.where( w_num <=0, 0., ratio) # Set to 0 if the numerator is negative for variations
            ratio_err =  np.where( (w_denom>0)&(w_num>0),
                                np.sqrt((1/w_denom)**2 * var_num + (w_num/w_denom**2)**2 * var_denom),
                                0.)
            ratio_err = np.where( w_num <=0, 0., ratio_err) # Set to 0 if the numerator is negative
            stack_ratio.append(ratio)
            stack_ratio_err.append(ratio_err)

        stack_ratio = np.stack(stack_ratio)
        stack_ratio_err = np.stack(stack_ratio_err)

        sfhist = hist.Hist(*h_num.axes, data=stack_ratio)
        sfhist_err = hist.Hist(*h_num.axes, data=stack_ratio_err)

        # Exporting it to correctionlib
        import correctionlib, rich
        import correctionlib.convert
        # without a name, the resulting object will fail validation
        sfhist.name = "ttlf_background_correction"
        sfhist.label = "out"
        clibcorr = correctionlib.convert.from_histogram(sfhist, flow="clamp")
        clibcorr.description = "SF to correct the tt+LF background to data substracted by other backgrounds"

        cset = correctionlib.schemav2.CorrectionSet(
            schema_version=2,
            description="tt+LF background correction",
            corrections=[clibcorr],
        )
        rich.print(cset)

        with open(f"{args.output}/ttlf_background_correction_{year}.json", "w") as fout:
            fout.write(cset.json(exclude_unset=True))

        if not args.no_plots:
            for variation in variations:
                print(f"Plotting the SF for {year}, variation {variation}")
                # Plot without numbers
                fig, (ax, ay) = plt.subplots(1, 2, figsize=(18, 7), dpi=dpi)
                #plt.subplots_adjust(wspace=0.3)

                ax.set_title(f"{year} {variation}")
                I = hep.hist2dplot(sfhist[{'variation':variation}], ax=ax, cmap=cmap, cbarextend=True, norm=Normalize(0.0, 1.5, clip=True))
                ax.set_yscale("log")
                ax.set_ylim(100, 5000)
                ax.set_xlabel("$N_{Jet}$")
                ax.set_ylabel("Jets $H_T$ [GeV]")
                ax.set_xticks(sfhist.axes[1].edges)
                #ax.set_yticks(sfhist.axes[2].edges)

                ay.set_title("stat. error")
                I = hep.hist2dplot(sfhist_err[{'variation':variation}], ax=ay, cmap=cmap, cbarextend=True, norm=Normalize(0.0, 0.5, clip=True))
                ay.set_yscale("log")
                ay.set_ylim(100, 5000)
                ay.set_xlabel("$N_{Jet}$")
                ay.set_ylabel("Jets $H_T$ [GeV]")
                ay.set_xticks(sfhist.axes[1].edges)
                #ay.set_yticks(sfhist.axes[2].edges)

                filename = f"{args.output}/plot_SFoutput_{year}_{variation}.png"
                print("Saving to", filename)
                print("Saving to", filename.replace(".png", ".pdf"))
                fig.savefig(filename)
                fig.savefig(filename.replace(".png", ".pdf"))
                plt.close(fig)

                # Plot with numbers
                fig, (ax, ay) = plt.subplots(1, 2, figsize=(18, 7), dpi=dpi)
                #plt.subplots_adjust(wspace=0.3)

                ax.set_title(f"{year} {variation}")
                I = hep.hist2dplot(sfhist[{'variation':variation}], ax=ax, cmap=cmap, cbarextend=True, norm=Normalize(0.0, 1.5, clip=True))
                ax.set_yscale("log")
                ax.set_ylim(100, 5000)
                ax.set_xlabel("$N_{Jet}$")
                ax.set_ylabel("Jets $H_T$ [GeV]")
                ax.set_xticks(sfhist.axes[1].edges)
                #ax.set_yticks(sfhist.axes[2].edges)
                annotate_bins(ax, sfhist[{'variation':variation}])

                ay.set_title("stat. error")
                I = hep.hist2dplot(sfhist_err[{'variation':variation}], ax=ay, cmap=cmap, cbarextend=True, norm=Normalize(0.0, 0.5, clip=True))
                ay.set_yscale("log")
                ay.set_ylim(100, 5000)
                ay.set_xlabel("$N_{Jet}$")
                ay.set_ylabel("Jets $H_T$ [GeV]")
                ay.set_xticks(sfhist.axes[1].edges)
                #ay.set_yticks(sfhist.axes[2].edges)
                annotate_bins(ay, sfhist_err[{'variation':variation}])

                filename = f"{args.output}/plot_SFoutput_{year}_{variation}_with_numbers.png"
                print("Saving to", filename)
                print("Saving to", filename.replace(".png", ".pdf"))
                fig.savefig(filename)
                fig.savefig(filename.replace(".png", ".pdf"))
                plt.close(fig)


######################################################
if args.validate and not args.no_plots:
    # Plot the shape with validation
    for var, sample, year in product(variables_to_plot, samples, years):
        print(f"Plotting validation for {var} {sample} {year}")
        shapes = [
            #(sample, cat, year, "nominal", "no tt+LF calibration"),
            (sample, cat, year, "nominal", "nominal"),
            (sample, cat, year, "sf_ttlf_calib_with_ttcc_variations_norm_ttccUp", r"ttcc 50% up"),
            (sample, cat, year, "sf_ttlf_calib_with_ttcc_variations_norm_ttccDown", r"ttcc 50% down"),
        ]
        os.makedirs(os.path.join(args.output, var), exist_ok=True)
        plot_shapes_comparison(output_variables, f"{var}", shapes, ylog=True,
                               lumi_label=f"{sample} {year}",
                               outputfile=os.path.join(args.output, f"{var}/{var}_{sample}_{year}.png"))

