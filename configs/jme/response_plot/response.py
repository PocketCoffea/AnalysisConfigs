# execute on slurm
# sbatch -p short --account=t3 --mem 10gb --cpus-per-task=16 --wrap="python response.py --full -d /work/mmalucch/out_jme/out_cartesian_full_recoEtaBins_CorrectJetPt_correctNeutrinosSeparation_jetpt_ZerosPtResponse_2023postBPix/  --histo -n 10"
# sbatch -p short --account=t3 --time=00:10:00 --mem 15gb --cpus-per-task=32 --wrap="python response.py --full -d /work/mmalucch/out_jme/out_cartesian_full_recoEtaBins_CorrectJetPt_correctNeutrinosSeparation_jetpt_ZerosPtResponse_newCorr_noReg_solved_numPrecision_allFits_2023_postBPix_closure/  --histo -n 32"

import os

os.environ["SIGN"] = ""

import sys
from coffea.util import load
import numpy as np
import matplotlib.pyplot as plt
import argparse
import mplhep as hep
from multiprocessing import Pool
import multiprocessing as mpr
from functools import partial
import functools
import json
from scipy.optimize import curve_fit
import scipy.stats as stats
import ROOT

ROOT.gROOT.SetBatch(True)


from pol_functions import *
from write_l2rel import write_l2rel_txt
from confidence import *
from histograms_to_plot import *

sys.path.append("../")
from params.binning import *

hep.style.use("CMS")

parser = argparse.ArgumentParser(description="Run the jme analysis")
parser.add_argument(
    "-u",
    "--unbinned",
    help="Binned or unbinned",
    action="store_true",
    default=False,
)
parser.add_argument(
    "-l",
    "--load",
    action="store_true",
    help="Load medians from file",
    default=False,
)
parser.add_argument(
    "-d",
    "--dir",
    type=str,
    help="Input dir",
)
parser.add_argument(
    "--cartesian",
    action="store_true",
    help="Run cartesian multicuts",
    default=False,
)
parser.add_argument(
    "--histo",
    action="store_true",
    help="Plot the histograms",
    default=False,
)
parser.add_argument(
    "--full",
    action="store_true",
    help="Run full cartesian analysis in all eta bins and all flavours sequentially",
    default=False,
)
parser.add_argument(
    "--test",
    action="store_true",
    help="Run test",
    default=False,
)
parser.add_argument(
    "--central",
    action="store_true",
    help="Run central eta bin",
    default=False,
)
parser.add_argument(
    "-a",
    "--abs-eta-inclusive",
    action="store_true",
    help="Run over inclusive abs eta bins",
    default=False,
)
parser.add_argument(
    "-n",
    "--num-processes",
    type=int,
    help="Number of processes",
    default=16,
)
parser.add_argument(
    "-p",
    "--num-params",
    type=int,
    help="Num param fit polynomial + 2 for the jet pt range",
    default=9,
)
parser.add_argument(
    "--no-plot",
    action="store_true",
    help="Do not plot",
    default=False,
)
parser.add_argument(
    "-c",
    "--choose-plot",
    action="store_true",
    help="Choose the plot",
    default=False,
)
parser.add_argument(
    "--flav",
    help="Flavour",
    type=str,
    default="inclusive",
)
parser.add_argument(
    "--all-flavs",
    help="Do all flavours",
    action="store_true",
    default=False,
)
args = parser.parse_args()


# set global variables
GOOD_FIT = 0
BAD_FIT = 0
VALID_FIT = 0
TOTAL_FIT = 0
DP_NOTE_PLOTS = True
REBIN = True
PLOT_SINGLE_HISTO = False
JET_PT = True
PLOT_JETPT_HISTO = False
PLOT_JETPT_MEDIAN = False
FIT = True
CLOSURE = False
HISTO_LOG = False
DENSITY = False
PDF = False
VERSION = "V3"


if "closure" in args.dir:
    FIT = False
    CLOSURE = True

# save the log also in a file
# sys.stdout = open(file=f"{args.dir}/response_plot.log", mode="w")
# sys.stderr = open(file=f"{args.dir}/response_plot.err", mode="w")

if "preEE" in args.dir:
    year = "Summer22Run3"
    year_txt = "Summer22_22Sep2023"
elif "postEE" in args.dir:
    year = "Summer22EERun3"
    year_txt = "Summer22EE_22Sep2023"
elif "preBPix" in args.dir:
    year = "Summer23Run3"
    year_txt = "Summer23Prompt23"
elif "postBPix" in args.dir:
    year = "Summer23BPixRun3"
    year_txt = "Summer23BPixPrompt23"

year_txt = year
if DP_NOTE_PLOTS:
    year = "2023" if "23" in year else "2022"


pt_bins = pt_bins_all if "pnetreg15" in args.dir else pt_bins_reduced


localdir = os.path.dirname(os.path.abspath(__file__))

if args.full and (args.central or args.abs_eta_inclusive or args.all_flavs):
    flavs = {
        ("inclusive",): ["o"],
        ("b", "c"): ["o", "x"],
        ("uds", "g"): ["o", "x"],
    }
elif args.full:
    flavs = {(args.flav,): ["o"]}


if "splitpnetreg15" in args.dir:
    # set color and marker for each variable
    variables_plot_settings = {
        "ResponseJEC": ["darkorange", "o"],
        "ResponseJECNeutrino": ["darkorange", "o"],
        "ResponseRaw": ["green", "s"],
        "ResponsePNetReg": ["red", "v"],
        "ResponsePNetRegSplit15": ["hotpink", "x"],
        "ResponsePNetRegTot": ["darkred", "<"],
        "ResponsePNetRegNeutrino": ["royalblue", "^"],
        "ResponsePNetRegNeutrinoSplit15": ["deepskyblue", "D"],
        "ResponsePNetRegNeutrinoTot": ["darkblue", ">"],
    }
    labels_dict = {
        "JEC": "JEC",
        "JECNeutrino": "JEC",
        "Raw": "Raw",
        "PNetReg": r"PNet (${p_{\text{T}}}^{\text{reco}}_{\text{raw}}$>15 GeV)",
        "PNetRegNeutrino": r"PNet incl. neutrinos (${p_{\text{T}}}^{\text{reco}}_{\text{raw}}$>15 GeV)",
        "PNetRegSplit15": r"PNet (${p_{\text{T}}}^{\text{reco}}_{\text{raw}}$<15 GeV)",
        "PNetRegNeutrinoSplit15": r"PNet incl. neutrinos (${p_{\text{T}}}^{\text{reco}}_{\text{raw}}$<15 GeV)",
        "PNetRegTot": "PNet",
        "PNetRegNeutrinoTot": "PNet incl. neutrinos",
    }
else:
    # set color and marker for each variable
    variables_plot_settings = {
        "ResponseJEC": ["darkorange", "o"],
        "ResponseJECNeutrino": ["darkorange", "o"],
        "ResponseRaw": ["green", "s"],
        "ResponsePNetReg": ["darkred", "<"],
        "ResponsePNetRegNeutrino": ["darkblue", ">"],
    }
    labels_dict = {
        "JEC": "JEC",
        "JECNeutrino": "JEC",
        "Raw": "Raw",
        "PNetReg": "PNet",
        "PNetRegNeutrino": "PNet incl. neutrinos",
    }

if JET_PT:
    # update the dictionary but using a for loop
    for key in list(variables_plot_settings.keys()):
        new_key = key.replace("Response", "JetPt")
        variables_plot_settings.update({new_key: variables_plot_settings[key]})

main_dir = args.dir.rstrip("/")
print("main_dir", main_dir)

if True:
    median_dir = (
        f"{main_dir}/median_plots_unbinned"
        if args.unbinned
        else f"{main_dir}/median_plots_binned"
    )
    os.makedirs(f"{median_dir}", exist_ok=True)
    print("median_dir", median_dir)
    inv_median_dir = (
        f"{main_dir}/inv_median_plots_unbinned"
        if args.unbinned
        else f"{main_dir}/inv_median_plots_binned"
    )
    os.makedirs(f"{inv_median_dir}", exist_ok=True)
    print("inv_median_dir", inv_median_dir)
    resolution_dir = (
        f"{main_dir}/resolution_plots_unbinned"
        if args.unbinned
        else f"{main_dir}/resolution_plots_binned"
    )
    os.makedirs(f"{resolution_dir}", exist_ok=True)
    print("resolution_dir", resolution_dir)
    weighted_resolution_dir = (
        f"{main_dir}/weighted_resolution_plots_unbinned"
        if args.unbinned
        else f"{main_dir}/weighted_resolution_plots_binned"
    )
    os.makedirs(f"{weighted_resolution_dir}", exist_ok=True)
    print("weighted_resolution_dir", weighted_resolution_dir)
    width_dir = (
        f"{main_dir}/width_plots_unbinned"
        if args.unbinned
        else f"{main_dir}/width_plots_binned"
    )
    os.makedirs(f"{width_dir}", exist_ok=True)
    print("width_dir", width_dir)

    if args.histo:
        histogram_dir = (
            f"{main_dir}/histogram_plots_unbinned"
            if args.unbinned
            else f"{main_dir}/histogram_plots_binned"
        )
        os.makedirs(f"{histogram_dir}", exist_ok=True)
        print("histogram_dir", histogram_dir)

plot_2d_dir = f"{main_dir}/2d_plots"
os.makedirs(f"{plot_2d_dir}", exist_ok=True)
os.makedirs(
    f"{plot_2d_dir}",
    exist_ok=True,
)

eta_sections = (
    list(eta_sign_dict.keys())
    if args.full
    else [main_dir.split("/")[-1].split("eta")[0]]
)
if args.central:
    eta_sections = ["central"]
elif args.abs_eta_inclusive:
    eta_sections = ["absinclusive"]
print("eta_sections", eta_sections)

for eta_sign, eta_interval in eta_sign_dict.items():
    if len(eta_sections) == 1 and eta_sections[0] == eta_sign:
        eta_bins = [
            i for i in eta_bins if i >= eta_interval[0] and i <= eta_interval[1]
        ]
        break
print("eta_bins", eta_bins)
correct_eta_bins = eta_bins

rebin_factors = {
    tuple(range(0, 8)): 60,
    tuple(range(8, 14)): 50,
    tuple(np.arange(14, 19)): 30,
    tuple(np.arange(19, 21)): 25,
    tuple(np.arange(21, 27)): 20,
}


def get_info_from_histogram(
    bins,
    values,
    variable,
    i,
    j,
    categories,
    medians_dict_el,
    err_medians_dict_el,
    resolutions_dict_el,
    width_dict_el,
    histogram_dict_el,
):
    bins_mid = (bins[1:] + bins[:-1]) / 2

    cdf = np.cumsum(values)
    cdf_normalized = cdf / cdf[-1]
    median_bin_index = np.argmax(cdf_normalized >= 0.5)
    median = bins_mid[median_bin_index]
    if "PNetReg" in variable and "4.889" not in categories[i]:
        condition = median < 0.1  # 0.8
    elif "PNetReg" in variable and "4.889" in categories[i]:
        condition = median < 0.1
    else:
        condition = False

    if np.sum(values) < 20 or (condition):
        # for k in range(
        #     j, len(h.axes[jet_pt])
        # ):
        # print(f"all values are 0 for eta {categories[i]} and pt {h.axes[jet_pt][j]}")
        medians_dict_el[variable][i].append(np.nan)
        err_medians_dict_el[variable][i].append(np.nan)
        resolutions_dict_el[variable][i].append(np.nan)
        width_dict_el[variable][i].append(np.nan)
        histogram_dict_el[variable][i].append(([], []))
        return

    if True:

        if REBIN:
            # rebin the histogram
            max_value = 2 if "Response" in variable else pt_bins[j + 1] * 2
            min_value = (
                # 1e-3
                0
                if "Response" in variable
                else pt_bins[j] * 0.5
            )
            try:
                max_index = np.where(bins > max_value)[0][0]
            except IndexError:
                max_index = len(bins) - 1
            # HERE
            # min_index = np.where(
            #     bins < min_value
            # )[0][-1]
            # min_index=0
            rebin_res = 1
            for d in rebin_factors.keys():
                if j in d:
                    rebin_res = rebin_factors[d]
            rebin_factor = (
                rebin_res
                if "Response" in variable
                else max(
                    len(
                        bins[
                            :max_index
                            # min_index:max_index
                        ]
                    )
                    // 60,
                    1,
                )
            )
            rebinned_bins = np.array(
                bins[:max_index][
                    # bins[min_index:max_index][
                    ::rebin_factor
                ]
            )
            rebinned_values = np.add.reduceat(
                values[:max_index],
                # values[min_index:max_index],
                range(
                    0,
                    len(
                        values[
                            :max_index
                            # min_index:max_index
                        ]
                    ),
                    rebin_factor,
                ),
            )
        else:
            rebinned_bins = bins
            rebinned_values = values

        rebinned_bins = list(rebinned_bins)
        rebinned_values = list(rebinned_values)

    if args.histo:
        histogram_dict_el[variable][i].append((rebinned_values, rebinned_bins))
    if "Response" in variable:

        # bins_mid = bins_mid[values != 0.]
        # values = values[values != 0.]
        # get the bins of the histo1d
        # find the bin which is the median of the histogram
        cdf = np.cumsum(values)
        # # print("cdf", cdf)
        cdf_normalized = cdf / cdf[-1]
        # # print("cdf_normalized", cdf_normalized)
        median_bin_index = np.argmax(cdf_normalized >= 0.5)
        # # print("median_bin_index", median_bin_index)
        median = bins_mid[median_bin_index]
        medians_dict_el[variable][i].append(median)

        mean = np.average(bins_mid, weights=values)
        rms = np.sqrt(
            np.average(
                (bins_mid - mean) ** 2,
                weights=values,
            )
        )
        err_median = 1.253 * rms / np.sqrt(np.sum(values))

        err_medians_dict_el[variable][i].append(err_median)

        values_noZero = values
        bins_mid_noZero = bins_mid

        if False:
            # get the index of the bin for which bins>=0.5
            # to remove low response bins for jec and raw respone
            index_05 = np.argmax(bins_mid >= 0.01)
            values_noZero = values[index_05:]
            bins_mid_noZero = bins_mid[index_05:]

        cdf_noZero = np.cumsum(values_noZero)
        cdf_normalized_noZero = cdf_noZero / cdf_noZero[-1]
        bin_width = bins_mid_noZero[1] - bins_mid_noZero[0]

        # compute standard resolution

        if False:
            # unbinned version
            width = Confidence_numpy(
                values_noZero,
                bins_mid_noZero,
                bin_width,
            )
        # binned version
        width = Confidence_numpy(
            rebinned_values,
            rebinned_bins,
            rebinned_bins[1] - rebinned_bins[0],
        )

        width_dict_el[variable][i].append(width)

        # define the resolution as the difference between the 84th and 16th percentile
        # find the bin which is the 84th percentile of the histogram
        percentile_84_bin_index = np.argmax(cdf_normalized_noZero >= 0.84)
        percentile_84 = bins_mid_noZero[percentile_84_bin_index]
        # find the bin which is the 16th percentile of the histogram
        percentile_16_bin_index = np.argmax(cdf_normalized_noZero >= 0.16)
        percentile_16 = bins_mid_noZero[percentile_16_bin_index]
        resolution = (percentile_84 - percentile_16) / 2
        resolutions_dict_el[variable][i].append(resolution)
    elif "JetPt" in variable:
        # compute the mean of the distribution
        mean = np.average(bins_mid, weights=values)
        medians_dict_el[variable][i].append(mean)
        # error on the mean
        err_medians_dict_el[variable][i].append(
            np.sqrt(
                np.average(
                    (bins_mid - mean) ** 2,
                    weights=values,
                )
            )
            / np.sqrt(np.sum(values))
        )

        resolutions_dict_el[variable][i].append(np.nan)
        width_dict_el[variable][i].append(np.nan)


if args.load:
    # TODO: if args.choose file load only needed
    print("loading from file")
    medians_dict = dict()
    err_medians_dict = dict()
    resolutions_dict = dict()
    width_dict = dict()
    if args.histo:
        histogram_dict = dict()
    for eta_sign in eta_sections:
        medians_dict[eta_sign] = dict()
        err_medians_dict[eta_sign] = dict()
        resolutions_dict[eta_sign] = dict()
        width_dict[eta_sign] = dict()
        if args.histo:
            histogram_dict[eta_sign] = dict()
        for flav_group in flavs:
            medians_dict[eta_sign][flav_group] = dict()
            err_medians_dict[eta_sign][flav_group] = dict()
            resolutions_dict[eta_sign][flav_group] = dict()
            width_dict[eta_sign][flav_group] = dict()
            if args.histo:
                histogram_dict[eta_sign][flav_group] = dict()
            for flav in flav_group:
                medians_dict[eta_sign][flav_group][flav] = dict()
                err_medians_dict[eta_sign][flav_group][flav] = dict()
                resolutions_dict[eta_sign][flav_group][flav] = dict()
                width_dict[eta_sign][flav_group][flav] = dict()
                if args.histo:
                    histogram_dict[eta_sign][flav_group][flav] = dict()
                for variable in list(variables_plot_settings.keys()):
                    print(
                        "eta_sign",
                        eta_sign,
                        "flav_group",
                        flav_group,
                        "flav",
                        flav,
                        "variable",
                        variable,
                    )
                    if args.full:
                        median_dir = (
                            f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/median_plots_unbinned"
                            if args.unbinned
                            else f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/median_plots_binned"
                        )
                        median_dir = (
                            f"{main_dir}/median_plots_unbinned"
                            if args.unbinned
                            else f"{main_dir}/median_plots_binned"
                        )
                        resolution_dir = (
                            f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/resolution_plots_unbinned"
                            if args.unbinned
                            else f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/resolution_plots_binned"
                        )
                        resolution_dir = (
                            f"{main_dir}/resolution_plots_unbinned"
                            if args.unbinned
                            else f"{main_dir}/resolution_plots_binned"
                        )
                        width_dir = (
                            f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/width_plots_unbinned"
                            if args.unbinned
                            else f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/width_plots_binned"
                        )
                        width_dir = (
                            f"{main_dir}/width_plots_unbinned"
                            if args.unbinned
                            else f"{main_dir}/width_plots_binned"
                        )
                        inv_median_dir = (
                            f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/inv_median_plots_unbinned"
                            if args.unbinned
                            else f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/inv_median_plots_binned"
                        )
                        inv_median_dir = (
                            f"{main_dir}/inv_median_plots_unbinned"
                            if args.unbinned
                            else f"{main_dir}/inv_median_plots_binned"
                        )
                        weighted_resolution_dir = (
                            f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/weighted_resolution_plots_unbinned"
                            if args.unbinned
                            else f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/weighted_resolution_plots_binned"
                        )
                        weighted_resolution_dir = (
                            f"{main_dir}/weighted_resolution_plots_unbinned"
                            if args.unbinned
                            else f"{main_dir}/weighted_resolution_plots_binned"
                        )
                        if args.histo:
                            histogram_dir = (
                                f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/histogram_plots_unbinned"
                                if args.unbinned
                                else f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/histogram_plots_binned"
                            )
                            histogram_dir = (
                                f"{main_dir}/histogram_plots_unbinned"
                                if args.unbinned
                                else f"{main_dir}/histogram_plots_binned"
                            )
                    try:
                        medians_dict[eta_sign][flav_group][flav][variable] = np.load(
                            f"{median_dir}/medians_{eta_sign}_{flav}_{variable}.npy"
                        )
                    except FileNotFoundError:
                        continue
                    try:
                        err_medians_dict[eta_sign][flav_group][flav][variable] = (
                            np.load(
                                f"{median_dir}/err_medians_{eta_sign}_{flav}_{variable}.npy"
                            )
                        )
                    except FileNotFoundError:
                        continue
                    try:
                        resolutions_dict[eta_sign][flav_group][flav][variable] = (
                            np.load(
                                f"{resolution_dir}/resolution_{eta_sign}_{flav}_{variable}.npy"
                            )
                        )
                    except FileNotFoundError:
                        continue
                    try:
                        width_dict[eta_sign][flav_group][flav][variable] = np.load(
                            f"{width_dir}/width_{eta_sign}_{flav}_{variable}.npy"
                        )
                    except FileNotFoundError:
                        continue

                    if args.histo:
                        # load from json files
                        try:
                            with open(
                                f"{histogram_dir}/histogram_{eta_sign}_{flav}_{variable}.json"
                            ) as f:
                                histogram_dict[eta_sign][flav_group][flav][variable] = (
                                    json.load(f)
                                )
                        except FileNotFoundError:
                            continue

    # else:
    #     medians_dict = np.load(f"{median_dir}/medians.npy")
    #     err_medians_dict = np.load(f"{median_dir}/err_medians.npy")
    # print("loaded", medians_dict, err_medians_dict)
else:
    if args.unbinned:
        print("unbinned")
        medians_dict = np.zeros((len(correct_eta_bins) - 1, len(pt_bins) - 1))
        err_medians_dict = np.zeros((len(correct_eta_bins) - 1, len(pt_bins) - 1))
        o_cartesian = load(f"{main_dir}/output_all.coffea") if args.cartesian else None
        num_tot = 0
        for i in range(len(correct_eta_bins) - 1):
            file_name = f"{main_dir}/eta{correct_eta_bins[i]}to{correct_eta_bins[i+1]}/output_all.coffea"
            o = load(file_name) if not args.cartesian else o_cartesian
            cat = (
                "baseline"
                if not args.cartesian
                else f"MatchedJets_eta{correct_eta_bins[i]}to{correct_eta_bins[i+1]}"
            )
            # histos_dict = o["variables"]
            columns_dict = o["columns"]
            # median_dict[f"eta{correct_eta_bins[i]}to{correct_eta_bins[i+1]}"] = {}
            for j in range(len(pt_bins) - 1):
                # var_histo = f"MatchedJets_Response_eta{correct_eta_bins[i]}to{correct_eta_bins[i+1]}_pt{pt_bins[j]}to{pt_bins[j+1]}"
                var = (
                    f"MatchedJets_eta{correct_eta_bins[i]}to{correct_eta_bins[i+1]}_pt{pt_bins[j]}to{pt_bins[j+1]}_Response"
                    if not args.cartesian
                    else f"MatchedJets_pt{pt_bins[j]}to{pt_bins[j+1]}_Response"
                )
                # print(f"variable {var} ")
                for sample in columns_dict.keys():
                    for dataset in columns_dict[sample].keys():
                        # print(f"variable {var} sample {sample} dataset {dataset}")
                        # histo = np.array(histos_dict[var_histo][sample][dataset][0, 0, :])["value"]
                        column = columns_dict[sample][dataset][cat][var].value
                        # sort column in ascending order
                        column = np.sort(column[column != -999.0])
                        median = np.median(column)
                        num_tot += len(column)
                        medians_dict[i, j] = median
                        # print("median:\n", median)

                        # compute the error on the median as 1.253 * hist->GetRMS() / TMath::Sqrt(hist->GetEffectiveEntries()); but for an unbinned distribution
                        mean = np.mean(column)
                        rms = np.sqrt(np.mean((column - mean) ** 2))
                        err_median = 1.253 * rms / np.sqrt(len(column))
                        err_medians_dict[i, j] = err_median
                        # print("err_median:\n", err_median)
        medians_dict = {"Response": medians_dict}
        err_medians_dict = {"Response": err_medians_dict}
    else:
        print("binned")
        # median_dict -> flav_group -> variable -> eta bins -> pt bins
        medians_dict = dict()
        err_medians_dict = dict()
        resolutions_dict = dict()
        width_dict = dict()
        histogram_dict = dict()
        values_tot_dict = dict()
        num_tot_dict = dict()

        o = load(f"{main_dir}/output_all.coffea") if not args.full else None
        variables = o["variables"].keys() if not args.full else None
        for eta_sign in eta_sections:
            medians_dict[eta_sign] = dict()
            err_medians_dict[eta_sign] = dict()
            resolutions_dict[eta_sign] = dict()
            width_dict[eta_sign] = dict()
            histogram_dict[eta_sign] = dict()
            values_tot_dict[eta_sign] = dict()
            num_tot_dict[eta_sign] = dict()

            for flav_group in flavs:
                medians_dict[eta_sign][flav_group] = dict()
                err_medians_dict[eta_sign][flav_group] = dict()
                resolutions_dict[eta_sign][flav_group] = dict()
                width_dict[eta_sign][flav_group] = dict()
                histogram_dict[eta_sign][flav_group] = dict()
                values_tot_dict[eta_sign][flav_group] = dict()
                num_tot_dict[eta_sign][flav_group] = dict()
                for flav in flav_group:
                    medians_dict[eta_sign][flav_group][flav] = dict()
                    err_medians_dict[eta_sign][flav_group][flav] = dict()
                    resolutions_dict[eta_sign][flav_group][flav] = dict()
                    width_dict[eta_sign][flav_group][flav] = dict()
                    histogram_dict[eta_sign][flav_group][flav] = dict()
                    values_tot_dict[eta_sign][flav_group][flav] = dict()
                    num_tot_dict[eta_sign][flav_group][flav] = dict()
                    for neutrino in ["", "_neutrino"]:
                        if args.full:
                            try:
                                o = load(
                                    f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/output_all{neutrino}.coffea"
                                )
                            except FileNotFoundError:
                                print(
                                    f"\n{main_dir}/{eta_sign}eta_{flav}flav_pnet/output_all{neutrino}.coffea not found\n"
                                )
                                continue
                            variables = o["variables"].keys()
                        for variable in variables:
                            if "VSpt" not in variable:
                                continue
                            histos_dict = o["variables"][variable]
                            # remove MatchedJets and VSpt from the name of the variable if present
                            variable = (
                                variable.replace(
                                    "MatchedJets_",
                                    "",  # if flav == "" else "MatchedJets", ""
                                ).replace("VSpt", "")
                                # .replace(flav, "")
                            )

                            medians_dict[eta_sign][flav_group][flav][variable] = list(
                                list()
                            )
                            err_medians_dict[eta_sign][flav_group][flav][variable] = (
                                list(list())
                            )
                            resolutions_dict[eta_sign][flav_group][flav][variable] = (
                                list(list())
                            )
                            width_dict[eta_sign][flav_group][flav][variable] = list(
                                list()
                            )
                            histogram_dict[eta_sign][flav_group][flav][variable] = list(
                                list()
                            )

                            tot_var_name = ""
                            if "splitpnetreg15" in args.dir and "PNet" in variable:
                                tot_var_name = variable.replace("Split15", "") + "Tot"
                                if (
                                    tot_var_name
                                    not in medians_dict[eta_sign][flav_group][
                                        flav
                                    ].keys()
                                ):
                                    medians_dict[eta_sign][flav_group][flav][
                                        tot_var_name
                                    ] = list(list())
                                    err_medians_dict[eta_sign][flav_group][flav][
                                        tot_var_name
                                    ] = list(list())
                                    resolutions_dict[eta_sign][flav_group][flav][
                                        tot_var_name
                                    ] = list(list())
                                    width_dict[eta_sign][flav_group][flav][
                                        tot_var_name
                                    ] = list(list())
                                    histogram_dict[eta_sign][flav_group][flav][
                                        tot_var_name
                                    ] = list(list())
                                    values_tot_dict[eta_sign][flav_group][flav][
                                        tot_var_name
                                    ] = list(list())
                                    num_tot_dict[eta_sign][flav_group][flav][
                                        tot_var_name
                                    ] = list(list())

                            for sample in histos_dict.keys():
                                for dataset in histos_dict[sample].keys():
                                    # histo = np.array(histos_dict[sample][dataset][cat][0, 0, :])["value"]
                                    histo = histos_dict[sample][dataset]
                                    # print(histo)
                                    categories = list(histo.axes["cat"])

                                    # remove the baseline category
                                    (
                                        categories.remove("baseline")
                                        if "baseline" in categories
                                        else None
                                    )

                                    # order the categories so that the ranges in eta are increasing
                                    categories = sorted(
                                        categories,
                                        key=lambda x: float(
                                            x.split("eta")[1].split("to")[0]
                                        ),
                                    )
                                    variations = list(histo.axes["variation"])
                                    lenght = len(categories) if not args.test else 1

                                    for i in range(lenght):
                                        medians_dict[eta_sign][flav_group][flav][
                                            variable
                                        ].append(list())
                                        err_medians_dict[eta_sign][flav_group][flav][
                                            variable
                                        ].append(list())
                                        resolutions_dict[eta_sign][flav_group][flav][
                                            variable
                                        ].append(list())
                                        width_dict[eta_sign][flav_group][flav][
                                            variable
                                        ].append(list())
                                        histogram_dict[eta_sign][flav_group][flav][
                                            variable
                                        ].append(list())

                                        if tot_var_name and (
                                            len(
                                                values_tot_dict[eta_sign][flav_group][
                                                    flav
                                                ][tot_var_name]
                                            )
                                            <= i
                                        ):
                                            medians_dict[eta_sign][flav_group][flav][
                                                tot_var_name
                                            ].append(list())
                                            err_medians_dict[eta_sign][flav_group][
                                                flav
                                            ][tot_var_name].append(list())
                                            resolutions_dict[eta_sign][flav_group][
                                                flav
                                            ][tot_var_name].append(list())
                                            width_dict[eta_sign][flav_group][flav][
                                                tot_var_name
                                            ].append(list())
                                            histogram_dict[eta_sign][flav_group][flav][
                                                tot_var_name
                                            ].append(list())
                                            values_tot_dict[eta_sign][flav_group][flav][
                                                tot_var_name
                                            ].append(list())
                                            num_tot_dict[eta_sign][flav_group][flav][
                                                tot_var_name
                                            ].append(list())

                                        for var in variations:
                                            h = histo[{"cat": categories[i]}][
                                                {"variation": var}
                                            ]
                                            # h is a histo2d and we want to find the median of the distribution along the axis MatchedJets.Response
                                            # for each bin in the axis MatchedJets.pt
                                            # so we need to loop over the bins in the axis MatchedJets.pt

                                            for jet_pt in [
                                                "MatchedJets.pt",
                                                "MatchedJetsNeutrino.pt",
                                                "MatchedJetsSplit15.pt",
                                                "MatchedJetsNeutrinoSplit15.pt",
                                            ]:
                                                try:
                                                    pt_axis_histo = h.axes[jet_pt]
                                                    break
                                                except KeyError:
                                                    continue

                                            for j in range(len(pt_axis_histo)):
                                                # print("eta_sign", eta_sign, "flav_group", flav_group, "flav", flav, "variable", variable, "eta", categories[i], "pt", h.axes[jet_pt][j])
                                                # get the histo1d for the bin j in the axis MatchedJets.pt
                                                h1d = h[{jet_pt: j}]
                                                # get the values of the histo1d
                                                values = h1d.values()
                                                bins = h1d.axes[0].edges

                                                # HERE
                                                # remove the first bin which is a peak to zero in the response
                                                bins = bins[2:]
                                                values = values[2:]

                                                # sum the values of the unbinned histogram
                                                if (
                                                    tot_var_name
                                                    and len(
                                                        values_tot_dict[eta_sign][
                                                            flav_group
                                                        ][flav][tot_var_name][i]
                                                    )
                                                    <= j
                                                ):
                                                    values_tot_dict[eta_sign][
                                                        flav_group
                                                    ][flav][tot_var_name][i].append(
                                                        np.array(values)
                                                    )
                                                    num_tot_dict[eta_sign][flav_group][
                                                        flav
                                                    ][tot_var_name][i].append(1)
                                                elif tot_var_name:
                                                    values_tot_dict[eta_sign][
                                                        flav_group
                                                    ][flav][tot_var_name][i][
                                                        j
                                                    ] += values
                                                    num_tot_dict[eta_sign][flav_group][
                                                        flav
                                                    ][tot_var_name][i][j] += 1

                                                get_info_from_histogram(
                                                    bins,
                                                    values,
                                                    variable,
                                                    i,
                                                    j,
                                                    categories,
                                                    medians_dict[eta_sign][flav_group][
                                                        flav
                                                    ],
                                                    err_medians_dict[eta_sign][
                                                        flav_group
                                                    ][flav],
                                                    resolutions_dict[eta_sign][
                                                        flav_group
                                                    ][flav],
                                                    width_dict[eta_sign][flav_group][
                                                        flav
                                                    ],
                                                    histogram_dict[eta_sign][
                                                        flav_group
                                                    ][flav],
                                                )

                                                if tot_var_name and (
                                                    num_tot_dict[eta_sign][flav_group][
                                                        flav
                                                    ][tot_var_name][i][j]
                                                    == 2
                                                ):
                                                    get_info_from_histogram(
                                                        bins,
                                                        values_tot_dict[eta_sign][
                                                            flav_group
                                                        ][flav][tot_var_name][i][j],
                                                        tot_var_name,
                                                        i,
                                                        j,
                                                        categories,
                                                        medians_dict[eta_sign][
                                                            flav_group
                                                        ][flav],
                                                        err_medians_dict[eta_sign][
                                                            flav_group
                                                        ][flav],
                                                        resolutions_dict[eta_sign][
                                                            flav_group
                                                        ][flav],
                                                        width_dict[eta_sign][
                                                            flav_group
                                                        ][flav],
                                                        histogram_dict[eta_sign][
                                                            flav_group
                                                        ][flav],
                                                    )

        for eta_sign in eta_sections:
            for flav_group in flavs:
                for flav in flav_group:
                    for variable in list(
                        medians_dict[eta_sign][flav_group][flav].keys()
                    ):
                        for dict_info in [
                            medians_dict,
                            err_medians_dict,
                            resolutions_dict,
                            width_dict,
                        ]:
                            dict_info[eta_sign][flav_group][flav][variable] = np.array(
                                dict_info[eta_sign][flav_group][flav][variable]
                            )

        if not args.full:
            correct_eta_bins = []
            for cat in categories:
                if "eta" not in cat:
                    continue
                print("cat", cat)
                # for each pair of eta bins, find the corresponding category
                eta_min = float(cat.split("eta")[1].split("to")[0])
                eta_max = float(cat.split("eta")[1].split("to")[1])
                # if abs(eta_min) > abs(eta_max):
                #     eta_min, eta_max = eta_max, eta_min
                if eta_min not in correct_eta_bins:
                    correct_eta_bins.append(eta_min)
                if eta_max not in correct_eta_bins:
                    correct_eta_bins.append(eta_max)

    for eta_sign in medians_dict.keys():
        for flav_group in medians_dict[eta_sign].keys():
            for flav in medians_dict[eta_sign][flav_group].keys():
                if args.full:
                    median_dir = (
                        f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/median_plots_unbinned"
                        if args.unbinned
                        else f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/median_plots_binned"
                    )
                    median_dir = (
                        f"{main_dir}/median_plots_unbinned"
                        if args.unbinned
                        else f"{main_dir}/median_plots_binned"
                    )
                    os.makedirs(f"{median_dir}", exist_ok=True)
                    resolution_dir = (
                        f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/resolution_plots_unbinned"
                        if args.unbinned
                        else f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/resolution_plots_binned"
                    )
                    resolution_dir = (
                        f"{main_dir}/resolution_plots_unbinned"
                        if args.unbinned
                        else f"{main_dir}/resolution_plots_binned"
                    )
                    os.makedirs(f"{resolution_dir}", exist_ok=True)
                    width_dir = (
                        f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/width_plots_unbinned"
                        if args.unbinned
                        else f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/width_plots_binned"
                    )
                    width_dir = (
                        f"{main_dir}/width_plots_unbinned"
                        if args.unbinned
                        else f"{main_dir}/width_plots_binned"
                    )
                    os.makedirs(f"{width_dir}", exist_ok=True)

                    inv_median_dir = (
                        f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/inv_median_plots_unbinned"
                        if args.unbinned
                        else f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/inv_median_plots_binned"
                    )
                    inv_median_dir = (
                        f"{main_dir}/inv_median_plots_unbinned"
                        if args.unbinned
                        else f"{main_dir}/inv_median_plots_binned"
                    )
                    os.makedirs(f"{inv_median_dir}", exist_ok=True)
                    weighted_resolution_dir = (
                        f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/weighted_resolution_plots_unbinned"
                        if args.unbinned
                        else f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/weighted_resolution_plots_binned"
                    )
                    weighted_resolution_dir = (
                        f"{main_dir}/weighted_resolution_plots_unbinned"
                        if args.unbinned
                        else f"{main_dir}/weighted_resolution_plots_binned"
                    )
                    os.makedirs(f"{weighted_resolution_dir}", exist_ok=True)
                    if args.histo:
                        histogram_dir = (
                            f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/histogram_plots_unbinned"
                            if args.unbinned
                            else f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/histogram_plots_binned"
                        )
                        histogram_dir = (
                            f"{main_dir}/histogram_plots_unbinned"
                            if args.unbinned
                            else f"{main_dir}/histogram_plots_binned"
                        )
                        os.makedirs(f"{histogram_dir}", exist_ok=True)
                for variable in medians_dict[eta_sign][flav_group][flav].keys():
                    np.save(
                        f"{median_dir}/medians_{eta_sign}_{flav}_{variable}.npy",
                        medians_dict[eta_sign][flav_group][flav][variable],
                    )
                    np.save(
                        f"{median_dir}/err_medians_{eta_sign}_{flav}_{variable}.npy",
                        err_medians_dict[eta_sign][flav_group][flav][variable],
                    )
                    np.save(
                        f"{resolution_dir}/resolution_{eta_sign}_{flav}_{variable}.npy",
                        resolutions_dict[eta_sign][flav_group][flav][variable],
                    )
                    np.save(
                        f"{width_dir}/width_{eta_sign}_{flav}_{variable}.npy",
                        width_dict[eta_sign][flav_group][flav][variable],
                    )
                    # TODO: save inverse median and weighted resolution
                    if args.histo:

                        # save in a json file
                        with open(
                            f"{histogram_dir}/histogram_{eta_sign}_{flav}_{variable}.json",
                            "w",
                        ) as f:
                            json.dump(
                                histogram_dict[eta_sign][flav_group][flav][variable],
                                f,
                                indent=4,
                            )

if args.central:
    correct_eta_bins = central_bins
elif args.abs_eta_inclusive:
    correct_eta_bins = inclusive_bins

correct_eta_bins = np.array(correct_eta_bins)
print("correct_eta_bins", correct_eta_bins, len(correct_eta_bins))


def compute_index_eta(eta_bin):

    for eta_sign, eta_interval in eta_sign_dict.items():
        if eta_bin < len(correct_eta_bins[correct_eta_bins < eta_interval[1]]):
            index = eta_bin - len(correct_eta_bins[correct_eta_bins < eta_interval[0]])
            eta_sign = eta_sign
            break

    return index, eta_sign


def fit_inv_median_pol(ax, x, y, xerr, yerr, variable, y_pos, name_plot):
    # print("fit_inv_median pol", variable)

    # fit iteartively with a polynomial of increasing order
    # until the p-value is greater than 0.05
    max_p_value = -1
    popt_list = []
    pcov_list = []
    chi2_list = []
    ndof_list = []
    p_value_list = []

    for i, pol in pol_functions_dict.items():
        p_initial = [1.0] * (i + 1)
        func = pol

        if i + 1 >= len(x) or i + 3 > args.num_params:
            break
        param_bounds = ([-1000.0] * len(p_initial), [1000.0] * len(p_initial))
        popt, pcov = curve_fit(
            func,
            x,
            y,
            p0=p_initial,
            sigma=yerr,
            absolute_sigma=True,
            # bounds=param_bounds
        )

        # print chi2 and p-value on the plot
        chi2 = np.sum(((y - func(x, *popt)) / yerr) ** 2)
        ndof = len(x) - len(popt)
        p_value = 1 - stats.chi2.cdf(chi2, ndof)

        # print(
        #     "\n",
        #     name_plot,
        #     "\nx",
        #     x,
        #     "\ny",
        #     y,
        #     "\nyerr",
        #     yerr,
        #     "\npopt",
        #     popt,
        #     "\npcov",
        #     pcov,
        #     "\nchi2/ndof",
        #     chi2,
        #     "/",
        #     ndof,
        #     "p_value",
        #     p_value,
        #     "\npol",
        #     i,
        # )

        if p_value > max_p_value:
            max_p_value = p_value

        popt_list.append(popt)
        pcov_list.append(pcov)
        chi2_list.append(chi2)
        ndof_list.append(ndof)
        p_value_list.append(p_value)

        if p_value > 0.05:
            break

        # if not np.isnan(p_value) and p_value < 0.05: #CHANGE
        #     # if last element of the dict, return empty dict
        #     if i == list(pol_functions_dict.keys())[-1]:
        #         return {}
        #     continue
    if max_p_value > 1e-7:
        index = p_value_list.index(max_p_value)
    elif max_p_value == -1:
        return
    else:
        # get the element with the least chi2
        index = chi2_list.index(min(chi2_list))

    # plot the fit
    # x_fit = np.linspace(x[0], x[-1], 1000)
    x_fit = np.logspace(np.log10(x[0]), np.log10(x[-1]), 1000)
    y_fit = pol_functions_dict[list(pol_functions_dict.keys())[index]](
        x_fit, *popt_list[index]
    )
    ax.plot(
        x_fit,
        y_fit,
        color=variables_plot_settings[variable][0],
        linestyle="-",
        linewidth=0.7,
    )

    if not DP_NOTE_PLOTS:
        ax.text(
            0.98,
            0.6 + y_pos,
            f"pol{index}, $\chi^2$ / ndof = {chi2_list[index]:.1f} / {ndof_list[index]}"
            + f", p-value = {max_p_value:.3f}",
            horizontalalignment="right",
            verticalalignment="top",
            transform=ax.transAxes,
            color=variables_plot_settings[variable][0],
        )

    fit_results = {
        "x": list(x),
        "y": list(y),
        "yerr": list(yerr),
        "jet_pt": [x[0], x[-1]],
        "y_fit_range": [y_fit[0], y_fit[-1]],
        "parameters": list(popt_list[index]),
        "pol": index,
        "errors": list(np.sqrt(np.diag(pcov_list[index]))),
        "chi2": chi2_list[index],
        "ndof": ndof_list[index],
        "p_value": p_value_list[index],
    }

    # print("\n", name_plot, variable, "fit_results", fit_results)
    # if index == 0:
    #     for i in range(len(popt_list)):
    #         print(
    #             "\n",
    #             name_plot,
    #             "\nx",
    #             x,
    #             "\ny",
    #             y,
    #             "\nyerr",
    #             yerr,
    #             "\npopt",
    #             popt_list[i],
    #             "\npcov",
    #             pcov_list[i],
    #             "\nchi2/ndof",
    #             chi2_list[i],
    #             "/",
    #             ndof_list[i],
    #             "p_value {:.20f}".format(p_value_list[i]),
    #             "\npol",
    #             i,
    #         )

    return fit_results


def plot_median_resolution(eta_bin, plot_type):

    if not args.central and not args.abs_eta_inclusive and args.full:
        index, eta_sign = compute_index_eta(eta_bin)
    elif args.central:
        index = eta_bin
        eta_sign = "central"
    elif args.abs_eta_inclusive:
        index = eta_bin
        eta_sign = "absinclusive"
    else:
        index = eta_bin
        eta_sign = eta_sections[0]

    if "median" in plot_type or "jet_pt" in plot_type:
        plot_dict = medians_dict
        err_plot_dict = err_medians_dict
    elif "resolution" in plot_type:
        plot_dict = resolutions_dict
        err_plot_dict = None
    elif "width" in plot_type:
        plot_dict = width_dict
        err_plot_dict = None
    else:
        print("plot_type not valid")
        return

    for neutrino_func, neutrino_str in (
        zip(
            [lambda x: "Neutrino" in x, lambda x: "Neutrino" not in x],
            ["_neutrino", ""],
        )
        if ("resolution" in plot_type or "width" in plot_type)
        else zip([lambda x: True], [""])
    ):
        for flav_group in plot_dict[eta_sign].keys():
            # create a file to save a dictionary with the fit results
            tot_fit_results = dict()
            # print("plotting median", flav_group, "eta", eta_sign)
            if "median" in plot_type or "jet_pt" in plot_type:
                fig, ax = plt.subplots()
            else:
                fig, (ax, ax_ratio) = plt.subplots(
                    2,
                    1,
                    sharex=True,
                    gridspec_kw={"height_ratios": [2.5, 1]},
                )
                fig.tight_layout()
                ax_ratio

            hep.cms.lumitext(f"{year} (13.6 TeV)", ax=ax)
            hep.cms.text(text="Simulation\nPreliminary", loc=2, ax=ax)

            ax.text(
                0.05,
                0.75 if "median" in plot_type or "jet_pt" in plot_type else 0.7,
                r"anti-$k_{T}$ R=0.4 (PUPPI)"
                + (
                    "\nRegression Closure Test"
                    if (CLOSURE and not DP_NOTE_PLOTS)
                    else ""
                )
                + f"\n{correct_eta_bins[eta_bin]} <"
                + (
                    r"$\eta^{reco}$"
                    if not args.abs_eta_inclusive
                    else r"$|\eta^{reco}$|"
                )
                + f"< {correct_eta_bins[eta_bin+1]}",
                horizontalalignment="left",
                verticalalignment="top",
                transform=ax.transAxes,
                color="black",
            )

            j = 0
            plot = False
            max_value = 0
            min_value = 1000
            for flav in plot_dict[eta_sign][flav_group].keys():
                y_pos = 0
                non_nan_other_var = None
                for variable in plot_dict[eta_sign][flav_group][flav].keys():
                    if (
                        ("jet_pt" in plot_type and "JetPt" not in variable)
                        or ("jet_pt" not in plot_type and "Response" not in variable)
                        or (CLOSURE and "Raw" in variable)
                    ):
                        continue
                    if not neutrino_func(variable):
                        continue

                    plot_array = plot_dict[eta_sign][flav_group][flav][variable][
                        index, :
                    ]
                    if not non_nan_other_var:
                        non_nan_other_var = len(plot_array)
                    err_plot_array = (
                        err_plot_dict[eta_sign][flav_group][flav][variable][index, :]
                        if err_plot_dict is not None
                        else None
                    )

                    if "inverse" in plot_type:
                        err_plot_array = err_plot_array / (plot_array**2)
                        plot_array = 1 / plot_array
                    if "weighted" in plot_type:
                        plot_array = (
                            plot_array
                            * 2
                            / medians_dict[eta_sign][flav_group][flav][variable][
                                index, :
                            ]
                        )
                    if (
                        variable not in list(variables_plot_settings.keys())
                        or np.all(np.isnan(plot_array))
                        or (
                            "splitpnetreg15" in args.dir
                            and "Tot" not in variable
                            and "PNet" in variable
                        )
                    ):
                        continue

                    max_value = (
                        max(max_value, np.nanmax(plot_array))
                        if not np.all(np.isnan(plot_array))
                        else max_value
                    )
                    min_value = (
                        min(min_value, np.nanmin(plot_array))
                        if not np.all(np.isnan(plot_array))
                        else min_value
                    )

                    plot = True

                    if "width" in plot_type:
                        for q in range(len(plot_array) - 1, non_nan_other_var, -1):
                            plot_array[q] = np.nan
                        # if the last point is distant form the latter more than 2.5 times the latter, remove it
                        non_nan = 0
                        for q in range(len(plot_array) - 1, 0, -1):

                            if ~np.isnan(plot_array[q]) and q > non_nan:
                                if any(
                                    [
                                        (
                                            plot_array[q] > 2.5 * plot_array[q - k]
                                            or plot_array[q] < plot_array[q - k] / 2.5
                                        )
                                        for k in range(1, 4)
                                    ]
                                ):

                                    plot_array[q] = np.nan
                                else:
                                    non_nan = q
                        non_nan_other_var = non_nan
                    ax.errorbar(
                        (
                            pt_bins[1:]
                            if "inverse" not in plot_type
                            else plot_dict[eta_sign][flav_group][flav][
                                variable.replace("Response", "JetPt")
                            ][index, :]
                        ),
                        plot_array,
                        yerr=(err_plot_array),
                        label=f"{labels_dict[variable.replace('Response','').replace('JetPt','')]}"
                        # + (" Closure Test" if CLOSURE and "PNet" in variable else "")
                        + (f" ({flav})" if flav != "inclusive" else ""),
                        marker=(
                            variables_plot_settings[variable][1]
                            if flav == "inclusive"
                            else flavs[flav_group][j]
                        ),
                        color=variables_plot_settings[variable][0],
                        linestyle="None",
                    )

                    # Fit the inverse median
                    if "inverse" in plot_type and "PNet" in variable:
                        mask_nan = (
                            ~np.isnan(plot_array)
                            & ~np.isnan(err_plot_array)
                            & (
                                ~np.isnan(
                                    plot_dict[eta_sign][flav_group][flav][
                                        variable.replace("Response", "JetPt")
                                    ][index, :]
                                )
                            )
                        )
                        x = plot_dict[eta_sign][flav_group][flav][
                            variable.replace("Response", "JetPt")
                        ][index, :]
                        xerr = err_plot_dict[eta_sign][flav_group][flav][
                            variable.replace("Response", "JetPt")
                        ][index, :]
                        # pt-clipping
                        mask_clip = x > 1  # HERE 35
                        mask_tot = mask_nan & mask_clip
                        x = x[mask_tot]
                        xerr = xerr[mask_tot]
                        y = plot_array[mask_tot]
                        y_err = err_plot_array[mask_tot]

                        if FIT and (
                            ("splitpnetreg15" not in args.dir)
                            or ("splitpnetreg15" in args.dir and "Tot" in variable)
                        ):

                            fit_results = fit_inv_median_pol(
                                ax,
                                x,
                                y,
                                # xerr,
                                np.zeros(len(x)),
                                y_err,
                                variable,
                                y_pos,
                                f"{eta_sign} {flav} {correct_eta_bins[eta_bin]} ({index}) {variable}",
                            )
                            y_pos += -0.05
                            tot_fit_results[f"{flav}_{variable}"] = fit_results
                            if fit_results == {}:
                                print(
                                    f"fit failed {flav} {variable} {eta_sign} {correct_eta_bins[eta_bin]}"
                                )

                    if "ResponsePNetReg" in variable and (
                        "resolution" in plot_type or "width" in plot_type
                    ):
                        # plot ratio pnreg / jec
                        jec = (
                            plot_dict[eta_sign][flav_group][flav]["ResponseJEC"][
                                index, :
                            ]
                            if (plot_type == "resolution" or plot_type == "width")
                            else plot_dict[eta_sign][flav_group][flav]["ResponseJEC"][
                                index, :
                            ]
                            * 2
                            / medians_dict[eta_sign][flav_group][flav]["ResponseJEC"][
                                index, :
                            ]
                        )
                        gain_res = (jec - plot_array) / jec
                        ax_ratio.errorbar(
                            pt_bins[1:],
                            gain_res,
                            # label= f"{variable.replace('Response','')} / JEC ({flav.replace('_','') if flav != '' else 'inclusive'})",
                            marker=(
                                variables_plot_settings[variable][1]
                                if flav == "inclusive"
                                else flavs[flav_group][j]
                            ),
                            color=variables_plot_settings[variable][0],
                            linestyle="None",
                        )
                    if "median" in plot_type:
                        # plot line at 1 for the whole figure
                        ax.axhline(y=1, color="black", linestyle="--", linewidth=0.7)

                j += 1
            # if no variable is plotted, skip
            if plot == False:
                continue
            # check if plot_array is only nan or 0
            if not np.all(np.isnan(plot_array)) and not np.all(plot_array == 0):
                if "resolution" in plot_type or "width" in plot_type:
                    ax.set_ylim(top=1.9 * max_value, bottom=min_value / 1.1)
                elif CLOSURE and "median" in plot_type:
                    ax.set_ylim(top=1.05 * max_value, bottom=min_value / 1.01)
                else:
                    ax.set_ylim(top=1.2 * max_value, bottom=min_value / 1.1)
            if "inverse" in plot_type:
                ax.set_xlabel(r"$p_{T}^{reco}$ (GeV)", loc="right")
            elif "median" in plot_type or "jet_pt" in plot_type:
                ax.set_xlabel(r"$p_{T}^{ptcl}$ (GeV)", loc="right")
            else:
                ax_ratio.set_xlabel(r"$p_{T}^{ptcl}$ (GeV)", loc="right")

            if plot_type == "median":
                label_y = f"Median jet response"
            elif plot_type == "inverse_median":
                label_y = r"[Median jet response]$^{-1}$"
            elif plot_type == "resolution":
                label_y = r"$\frac{q_{84}-q_{16}}{2}$"
            elif plot_type == "weighted_resolution":
                label_y = r"$\frac{q_{84}-q_{16}}{q_{50}}$"
            elif plot_type == "width":
                label_y = "Jet energy resolution"
            elif plot_type == "average_jet_pt":
                # ax.set_yscale("log")
                label_y = r"$\langle p_{T}^{Jet} \rangle$ (GeV)"

            ax.set_ylabel(label_y, loc="top")

            # log x scale
            ax.set_xscale("log")

            # if not DP_NOTE_PLOTS:
            if tot_fit_results:
                old_xlim = ax.get_xlim()

                for name, value in tot_fit_results.items():
                    # draw the dashed lines to show the fit range
                    ax.plot(
                        [old_xlim[0], value["jet_pt"][0]],
                        [value["y_fit_range"][0], value["y_fit_range"][0]],
                        color=variables_plot_settings[name.split("_")[-1]][0],
                        linestyle="--",
                        linewidth=0.7,
                    )
                    ax.plot(
                        [value["jet_pt"][1], old_xlim[1]],
                        [value["y_fit_range"][1], value["y_fit_range"][1]],
                        color=variables_plot_settings[name.split("_")[-1]][0],
                        linestyle="--",
                        linewidth=0.7,
                    )
                ax.set_xlim(old_xlim)

            handles, labels = ax.get_legend_handles_labels()
            handles_dict = dict(zip(labels, handles))
            unique_labels = list((handles_dict.keys()))
            unique_dict = {label: handles_dict[label] for label in unique_labels}
            # add elemtns to the legend by hand

            # if plot_type == "inverse_median" and FIT:
            #     unique_dict["Fit standard+Gaussian"] = plt.Line2D(
            #         [0], [0], color="black", linestyle="--", label="Fit standard+Gaussian"
            #     )

            ax.legend(
                unique_dict.values(),
                unique_dict.keys(),
                frameon=False,
                ncol=1,
                loc="upper right",
            )

            # ax.*.grid(color="gray", linestyle=":", linewidth=0.4, which="both")
            if "resolution" in plot_type or "width" in plot_type:
                ax_ratio.set_ylabel("1 - PNet / JEC", loc="bottom")  # 1-PNet/Standard
                # ax.*.grid(color="gray", linestyle=":", linewidth=0.4, which="both")

            # create string for flavour
            flav_str = ""
            for flav in flav_group:
                flav_str += flav.replace("_", "")

            if plot_type == "median" or plot_type == "average_jet_pt":
                plots_dir = median_dir
            elif plot_type == "inverse_median":
                plots_dir = inv_median_dir
            elif plot_type == "resolution":
                plots_dir = resolution_dir
            elif plot_type == "weighted_resolution":
                plots_dir = weighted_resolution_dir
            elif plot_type == "width":
                plots_dir = width_dir

            if args.full:
                if plot_type == "median" or plot_type == "average_jet_pt":
                    plots_dir = [
                        (
                            f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/median_plots_unbinned"
                            if args.unbinned
                            else f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/median_plots_binned"
                        )
                        for flav in flav_group
                    ]
                    plots_dir = [
                        (
                            f"{main_dir}/median_plots_unbinned"
                            if args.unbinned
                            else f"{main_dir}/median_plots_binned"
                        )
                        for flav in flav_group
                    ]
                elif plot_type == "inverse_median":
                    plots_dir = [
                        (
                            f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/inv_median_plots_unbinned"
                            if args.unbinned
                            else f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/inv_median_plots_binned"
                        )
                        for flav in flav_group
                    ]
                    plots_dir = [
                        (
                            f"{main_dir}/inv_median_plots_unbinned"
                            if args.unbinned
                            else f"{main_dir}/inv_median_plots_binned"
                        )
                        for flav in flav_group
                    ]
                elif plot_type == "resolution":
                    plots_dir = [
                        (
                            f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/resolution_plots_unbinned"
                            if args.unbinned
                            else f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/resolution_plots_binned"
                        )
                        for flav in flav_group
                    ]
                    plots_dir = [
                        (
                            f"{main_dir}/resolution_plots_unbinned"
                            if args.unbinned
                            else f"{main_dir}/resolution_plots_binned"
                        )
                        for flav in flav_group
                    ]

                elif plot_type == "weighted_resolution":
                    plots_dir = [
                        (
                            f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/weighted_resolution_plots_unbinned"
                            if args.unbinned
                            else f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/weighted_resolution_plots_binned"
                        )
                        for flav in flav_group
                    ]

                    plots_dir = [
                        (
                            f"{main_dir}/weighted_resolution_plots_unbinned"
                            if args.unbinned
                            else f"{main_dir}/weighted_resolution_plots_binned"
                        )
                        for flav in flav_group
                    ]
                elif plot_type == "width":
                    plots_dir = [
                        (
                            f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/width_plots_unbinned"
                            if args.unbinned
                            else f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/width_plots_binned"
                        )
                        for flav in flav_group
                    ]
                    plots_dir = [
                        (
                            f"{main_dir}/width_plots_unbinned"
                            if args.unbinned
                            else f"{main_dir}/width_plots_binned"
                        )
                        for flav in flav_group
                    ]
                for plot_dir in plots_dir:
                    fig.savefig(
                        f"{plot_dir}/{plot_type}_{'JetPt' if 'jet_pt' in plot_type else 'Response'}_{flav_str}_{neutrino_str}_eta{correct_eta_bins[eta_bin]}to{correct_eta_bins[eta_bin+1]}.{'pdf' if PDF else 'png'}",
                        bbox_inches="tight",
                        dpi=300,
                    )
                    if tot_fit_results:
                        # save the fit results dictionary to a file
                        with open(
                            f"{plot_dir}/fit_results_{plot_type}_Response_{flav_str}_eta{correct_eta_bins[eta_bin]}to{correct_eta_bins[eta_bin+1]}.json",
                            "w",
                        ) as f:
                            json.dump(tot_fit_results, f, indent=4)

            else:
                fig.savefig(
                    f"{plots_dir}/{plot_type}_{'JetPt' if 'jet_pt' in plot_type else 'Response'}_{flav_str}_{neutrino_str}_eta{correct_eta_bins[eta_bin]}to{correct_eta_bins[eta_bin+1]}.{'pdf' if PDF else 'png'}",
                    bbox_inches="tight",
                    dpi=300,
                )
                if tot_fit_results:
                    # save the fit results dictionary to a file
                    with open(
                        f"{plots_dir}/fit_results_{plot_type}_Response_{flav_str}_eta{correct_eta_bins[eta_bin]}to{correct_eta_bins[eta_bin+1]}.json",
                        "w",
                    ) as f:
                        json.dump(tot_fit_results, f)
            plt.close(fig)


def plot_histos(eta_pt, histogram_dir):
    eta_bin = eta_pt[0]
    pt_bin = eta_pt[1]

    if not args.central and not args.abs_eta_inclusive and args.full:
        index, eta_sign = compute_index_eta(eta_bin)
    elif args.central:
        index = eta_bin
        eta_sign = "central"
    elif args.abs_eta_inclusive:
        index = eta_bin
        eta_sign = "absinclusive"
    else:
        index = eta_bin
        eta_sign = eta_sections[0]

    # for eta_sign in medians_dict.keys():
    for flav_group in histogram_dict[eta_sign].keys():
        if args.choose_plot and not any(
            [(flav_group == histo[1]) for histo in histograms_to_plot]
        ):
            continue
        # print("plotting histos", flav_group, "eta", eta_sign    )
        for flav in histogram_dict[eta_sign][flav_group].keys():
            if args.choose_plot and not any(
                [(flav == histo[2]) for histo in histograms_to_plot]
            ):
                continue
            plot_response = False
            plot_jetpt = False

            fig_tot_response, ax_tot_response = plt.subplots(
                figsize=((13, 13) if "splitpnetreg15" in args.dir else None)
            )
            fig_tot_jetpt, ax_tot_jetpt = plt.subplots(
                # figsize=((15, 9) if "splitpnetreg15" in args.dir else None)
            )

            max_value_response = 0
            max_value_jetpt = 0
            median = None
            resolution = None

            for variable in histogram_dict[eta_sign][flav_group][flav].keys():
                if args.choose_plot and not any(
                    [(variable == histo[3]) for histo in histograms_to_plot]
                ):
                    continue
                if "JetPt" in variable and not PLOT_JETPT_HISTO:
                    continue
                if variable not in list(variables_plot_settings.keys()):
                    # print(
                    #     "skipping",
                    #     variable,
                    #     "index",
                    #     index,
                    #     "eta_sign",
                    #     eta_sign,
                    #     flav_group,
                    #     flav,
                    # )
                    continue
                if (
                    "splitpnetreg15" in args.dir
                    and "PNetReg" in variable
                    and "Tot" not in variable
                    and pt_bin >= 9
                    and not args.choose_plot
                ):
                    continue
                if (
                    "splitpnetreg15" in args.dir
                    and "PNetReg" not in variable
                    and pt_bin < 9
                    and not args.choose_plot
                ):
                    continue

                histos = histogram_dict[eta_sign][flav_group][flav][variable]

                if args.full:
                    histogram_dir = (
                        f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/histogram_plots_unbinned"
                        if args.unbinned
                        else f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/histogram_plots_binned"
                    )
                    histogram_dir = (
                        f"{main_dir}/histogram_plots_unbinned"
                        if args.unbinned
                        else f"{main_dir}/histogram_plots_binned"
                    )
                    os.makedirs(f"{histogram_dir}", exist_ok=True)
                values = histos[index][pt_bin][0]
                bins = histos[index][pt_bin][1]
                if len(values) == 0:
                    # print(
                    #     "variable",
                    #     variable,
                    #     "values",
                    #     values,
                    #     "bins",
                    #     bins,
                    #     "pt_bin",
                    #     pt_bins[pt_bin],
                    #     "eta_bin",
                    #     eta_bin,
                    #     "is empty, skipping",
                    # )
                    continue
                if (
                    "PNetRegNeutrino" in variable
                    and (
                        ("splitpnetreg15" in args.dir and "Tot" in variable)
                        or "splitpnetreg15" not in args.dir
                    )
                    and not args.choose_plot
                ):
                    median = medians_dict[eta_sign][flav_group][flav][variable][index][
                        pt_bin
                    ]
                    resolution = resolutions_dict[eta_sign][flav_group][flav][variable][
                        index
                    ][pt_bin]
                    if np.isnan(median) or np.isnan(resolution):
                        median = None
                        resolution = None

                plot_response = True if "Response" in variable else plot_response
                plot_jetpt = True if "JetPt" in variable else plot_jetpt

                if "Response" in variable:
                    ax_tot_response.hist(
                        bins,
                        bins=bins,
                        weights=values,
                        histtype="step",
                        label=f'{labels_dict[variable.replace("Response", "")]}'
                        + (f" ({flav})" if flav != "inclusive" else ""),
                        color=variables_plot_settings[variable][0],
                        density=DENSITY,
                        linewidth=1.5,
                    )
                    max_value_response = max(
                        max_value_response,
                        np.nanmax(values)
                        / ((np.sum(values) * np.diff(bins)[0]) if DENSITY else 1),
                    )
                if "JetPt" in variable:
                    ax_tot_jetpt.hist(
                        bins,
                        bins=bins,
                        weights=values,
                        histtype="step",
                        label=f'{labels_dict[variable.replace("JetPt", "")]}'
                        + (f" ({flav})" if flav != "inclusive" else ""),
                        color=variables_plot_settings[variable][0],
                        density=DENSITY,
                        linewidth=1.5,
                    )
                    max_value_jetpt = max(
                        max_value_jetpt,
                        np.nanmax(values)
                        / ((np.sum(values) * np.diff(bins)[0]) if DENSITY else 1),
                    )

                    if False and "Raw" in variable:
                        tot_ev = np.sum(values)
                        ev_below8 = np.sum(np.array(values)[np.array(bins) <= 8])

                        print(
                            "tot_ev",
                            tot_ev,
                            "ev_below8",
                            ev_below8,
                            "ratio",
                            ev_below8 / tot_ev,
                        )

                if PLOT_SINGLE_HISTO:
                    fig, ax = plt.subplots(
                        # figsize=((16, 12) if "splitpnetreg15" in args.dir else None)
                    )
                    # bins_mid = (bins[1:] + bins[:-1]) / 2
                    # print("values", len(values), "bins", len(bins), "bins_mid", len(bins_mid))
                    ax.hist(
                        bins,
                        bins=bins,
                        weights=values,
                        histtype="step",
                        label=f'{labels_dict[variable.replace("Response", "").replace("JetPt", "")]}'
                        + (f" ({flav})" if flav != "inclusive" else ""),
                        color=variables_plot_settings[variable][0],
                        density=DENSITY,
                        linewidth=1.5,
                    )

                    # write axis name in latex
                    ax.set_xlabel(
                        (
                            r"$p_T^{reco} / p_T^{ptcl}$"
                            if "Response" in variable
                            else r"$p_{T}^{reco}$"
                        ),
                        loc="right",
                    )
                    ax.set_ylabel("a.u." if DENSITY else "Events", loc="top")
                    # if np.any(values != np.nan) and np.any(values != 0):
                    #     ax.set_ylim(top=1.2 * np.nanmax(values))

                    ax.legend(frameon=False, loc="upper right")

                    # ax.*.grid(color="gray", linestyle=":", linewidth=0.4, which="both")

                    # hep.cms.label(
                    #     year=year,
                    #     com="13.6",
                    #     label=f"Preliminary",
                    # )
                    hep.cms.lumitext(f"{year} (13.6 TeV)", ax=ax)
                    hep.cms.text(text="Simulation\nPreliminary", loc=2, ax=ax)
                    ax.text(
                        0.05,
                        0.75,
                        r"anti-$k_{T}$ R=0.4 (PUPPI)"
                        + f"\n{correct_eta_bins[eta_bin]} <"
                        + (
                            r"$\eta^{reco}$"
                            if not args.abs_eta_inclusive
                            else r"$|\eta^{reco}$|"
                        )
                        + f"< {correct_eta_bins[eta_bin+1]}\n"
                        + f"{int(pt_bins[pt_bin])} <"
                        + r"$p_{T}^{ptcl}$"
                        + f"< {int(pt_bins[pt_bin+1])}",
                        horizontalalignment="left",
                        verticalalignment="top",
                        transform=ax.transAxes,
                    )
                    if HISTO_LOG:
                        ax.set_yscale("log")
                    fig.savefig(
                        f"{histogram_dir}/histos_{variable}_{flav}_eta{correct_eta_bins[eta_bin]}to{correct_eta_bins[eta_bin+1]}_pt{pt_bins[pt_bin]}to{pt_bins[pt_bin+1]}{'_log' if HISTO_LOG else ''}.{'pdf' if PDF else 'png'}",
                        bbox_inches="tight",
                        dpi=300,
                    )
                    plt.close(fig)

            # check if the plot has plotted histograms
            if plot_response:
                # ax.grid(
                #     color="gray", linestyle=":", linewidth=0.4, which="both"
                # )
                # write axis name in latex
                ax_tot_response.set_xlabel(r"$p_T^{reco} / p_T^{ptcl}$", loc="right")
                ax_tot_response.set_ylabel("a.u." if DENSITY else "Events", loc="top")
                ax_tot_response.legend(frameon=False, loc="upper right", ncol=1)
                if max_value_response != 0 and not HISTO_LOG:
                    ax_tot_response.set_ylim(
                        top=(
                            1.2 * max_value_response
                            if not DP_NOTE_PLOTS
                            else (
                                1.8 * max_value_response
                                if not args.choose_plot
                                else 1.5 * max_value_response
                            )
                        )
                    )

                if not HISTO_LOG and not DP_NOTE_PLOTS:
                    if median and resolution:
                        ax_tot_response.set_xlim(
                            right=median + 4 * resolution, left=median - 4 * resolution
                        )
                    else:
                        ax_tot_response.set_xlim(right=1.8, left=0.5)
                elif not HISTO_LOG and DP_NOTE_PLOTS:
                    if median and resolution:
                        ax_tot_response.set_xlim(
                            right=median + 4 * resolution, left=median - 4 * resolution
                        )
                    else:
                        ax_tot_response.set_xlim(right=1.8, left=0.5)
                    # ax_tot_response.set_xlim(right=1.3, left=0.7)

                # hep.cms.label(
                #     year=year,
                #     com="13.6",
                #     label=f"Preliminary",
                #     ax=ax_tot_response,
                # )

                hep.cms.lumitext(f"{year} (13.6 TeV)", ax=ax_tot_response)
                hep.cms.text(text="Simulation\nPreliminary", loc=2, ax=ax_tot_response)

                ax_tot_response.text(
                    0.05,
                    0.75,
                    r"anti-$k_{T}$ R=0.4 (PUPPI)"
                    + f"\n{correct_eta_bins[eta_bin]} <"
                    + (
                        r"$\eta^{reco}$"
                        if not args.abs_eta_inclusive
                        else r"$|\eta^{reco}$|"
                    )
                    + f"< {correct_eta_bins[eta_bin+1]}\n"
                    + f"{int(pt_bins[pt_bin])} <"
                    + r"$p_{T}^{ptcl}$ (GeV)"
                    + f"< {int(pt_bins[pt_bin+1])}",
                    horizontalalignment="left",
                    verticalalignment="top",
                    transform=ax_tot_response.transAxes,
                    #
                )

                if HISTO_LOG:
                    ax_tot_response.set_yscale("log")
                fig_tot_response.savefig(
                    f"{histogram_dir}/histos_ResponseAll_{flav}_eta{correct_eta_bins[eta_bin]}to{correct_eta_bins[eta_bin+1]}_pt{pt_bins[pt_bin]}to{pt_bins[pt_bin+1]}{'_log' if HISTO_LOG else ''}.{'pdf' if PDF else 'png'}",
                    bbox_inches="tight",
                    dpi=300,
                )
            if plot_jetpt:
                # ax.grid(
                #     color="gray", linestyle=":", linewidth=0.4, which="both"
                # )
                ax_tot_jetpt.set_xlabel(r"$p_{T}^{reco}$", loc="right")
                ax_tot_jetpt.set_ylabel("a.u." if DENSITY else "Events", loc="top")
                ax_tot_jetpt.legend(frameon=False, loc="upper right", ncol=2)
                if max_value_jetpt != 0 and not HISTO_LOG:
                    ax_tot_jetpt.set_ylim(top=1.2 * max_value_jetpt)
                if not HISTO_LOG:
                    ax_tot_jetpt.set_xlim(
                        left=0.3 * pt_bins[pt_bin + 1], right=3 * pt_bins[pt_bin]
                    )

                # hep.cms.label(
                #     year=year, com="13.6", label=f"Preliminary", ax=ax_tot_jetpt
                # )

                hep.cms.lumitext(f"{year} (13.6 TeV)", ax=ax_tot_jetpt)
                hep.cms.text(text="Simulation\nPreliminary", loc=2, ax=ax_tot_jetpt)
                ax_tot_jetpt.text(
                    0.05,
                    0.75,
                    r"anti-$k_{T}$ R=0.4 (PUPPI)"
                    + f"\n{correct_eta_bins[eta_bin]} <"
                    + (
                        r"$\eta^{reco}$"
                        if not args.abs_eta_inclusive
                        else r"$|\eta^{reco}$|"
                    )
                    + f"< {correct_eta_bins[eta_bin+1]}\n"
                    + f"{int(pt_bins[pt_bin])} <"
                    + r"$p_{T}^{ptcl}$"
                    + f"< {int(pt_bins[pt_bin+1])}",
                    horizontalalignment="left",
                    verticalalignment="top",
                    transform=ax_tot_jetpt.transAxes,
                )
                if HISTO_LOG:
                    ax_tot_jetpt.set_yscale("log")

                fig_tot_jetpt.savefig(
                    f"{histogram_dir}/histos_JetPtAll_{flav}_eta{correct_eta_bins[eta_bin]}to{correct_eta_bins[eta_bin+1]}_pt{pt_bins[pt_bin]}to{pt_bins[pt_bin+1]}{'_log' if HISTO_LOG else ''}.{'pdf' if PDF else 'png'}",
                    bbox_inches="tight",
                    dpi=300,
                )
            plt.close(fig_tot_response)
            plt.close(fig_tot_jetpt)


def plot_2d(plot_dict, pt_bins_2d, correct_eta_bins_2d):
    # plot_dict["full"] = dict()
    # for eta_sign in plot_dict.keys():
    #     for flav_group in plot_dict[eta_sign].keys():
    #         plot_dict["full"][flav_group] = dict()
    #         print(eta_sign,  plot_dict[eta_sign][flav_group].keys())
    #         for flav in plot_dict[eta_sign][flav_group].keys():
    #             plot_dict["full"][flav_group][flav] = dict()
    #             print("full", plot_dict["full"][flav_group][flav].keys())
    #             for variable in plot_dict[eta_sign][flav_group][flav].keys():
    #                 length = len(plot_dict[eta_sign][flav_group][flav][variable])
    #                 # print("length", length)
    #                 plot_dict["full"][flav_group][flav][variable] = list(list())
    #                 for i in range(length):
    #                     plot_dict["full"][flav_group][flav][variable].append(
    #                         plot_dict[eta_sign][flav_group][flav][variable][i]
    #                     )

    # print("plot_dict[full]", plot_dict["full"])
    # print(plot_dict["full"][("inclusive",)]["inclusive"].keys())

    for eta_sign in plot_dict.keys():  # ["full"]:
        for flav_group in plot_dict[eta_sign].keys():
            # print("plotting median", flav_group, "eta", eta_sign)

            for flav in plot_dict[eta_sign][flav_group].keys():
                for variable in plot_dict[eta_sign][flav_group][flav].keys():
                    if variable not in list(variables_plot_settings.keys()):
                        continue
                    median_2d = plot_dict[eta_sign][flav_group][flav][variable]
                    median_2d = np.array(median_2d)  # -1

                    fig, ax = plt.subplots(
                        # figsize=((16, 12) if "splitpnetreg15" in args.dir else None)
                    )

                    hep.cms.lumitext(f"{year} (13.6 TeV)", ax=ax)
                    hep.cms.text(text="Simulation\nPreliminary", loc=2, ax=ax)
                    # print(
                    #     "plotting median",
                    #     flav_group,
                    #     flav,
                    #     variable,
                    #     "eta",
                    #     correct_eta_bins[i],
                    #     correct_eta_bins[i + 1],
                    #     "index",
                    #     index,
                    # )

                    # plot 2d
                    # print(median_2d)
                    # put zeros instead of nan
                    # median_2d[np.isnan(median_2d)] = 0
                    # print(median_2d)
                    len_eta = int((len(correct_eta_bins_2d) - 1) / len(eta_sections))
                    # require pt > 30 and < 1000
                    mask_pt = (pt_bins_2d >= 30) & (pt_bins_2d <= 3000)
                    # TODO: correct the eta signs
                    mask_eta = (
                        abs(correct_eta_bins_2d[: len_eta + 1]) < 4
                        if "neg" in eta_sign
                        else abs(correct_eta_bins_2d[len_eta:]) < 4
                    )

                    # print(pt_bins_2d, len(pt_bins_2d))
                    # print(correct_eta_bins_2d, len(correct_eta_bins_2d))
                    # print("h2d", median_2d[-1])
                    pt_bins_2d_cut = pt_bins_2d[mask_pt]
                    correct_eta_bins_2d_cut = (
                        correct_eta_bins_2d[: len_eta + 1][mask_eta]
                        if "neg" in eta_sign
                        else correct_eta_bins_2d[len_eta:][mask_eta]
                    )
                    median_2d = median_2d[mask_eta[:-1], :]
                    # print("h2d0", median_2d[-1], len(median_2d),   len(median_2d[0]))
                    median_2d = median_2d[:, mask_pt[:-1]][:, :-1]
                    if "pos" in eta_sign:
                        median_2d = median_2d[:-1, :]
                    # print(pt_bins_2d, len(pt_bins_2d))
                    # print(correct_eta_bins_2d, len(correct_eta_bins_2d))
                    # print("h2d1", median_2d[-1], len(median_2d),   len(median_2d[0]))
                    c = plt.pcolormesh(
                        pt_bins_2d_cut,
                        correct_eta_bins_2d_cut,
                        median_2d,
                        cmap="viridis",
                        # norm=LogNorm(vmin=0.0001, vmax=1),
                        # vmin=0.95,
                        # vmax=1.05,
                        # label=f"{variable.replace('Response','')} ({flav.replace('_','') if flav != '' else 'inclusive'})",
                    )
                    plt.colorbar(c)
                    ax.text(
                        0.98,
                        0.8 if "pos" in eta_sign else 0.2,
                        f"Median jet response\n{variable.replace('Response','')}"
                        + (f" ({flav})" if flav != "inclusive" else ""),
                        horizontalalignment="right",
                        verticalalignment="top",
                        transform=ax.transAxes,
                        bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9),
                    )

                    ax.set_xlabel(r"$p_{T}^{ptcl}$ (GeV)", loc="right")
                    ax.set_ylabel(r"$\eta^{reco}$", loc="top")
                    # ax.*.grid(color="gray", linestyle=":", linewidth=0.4, which="both")

                    # ax.legend(frameon=False, ncol=2, loc="upper right")

                    fig.savefig(
                        f"{plot_2d_dir}/median_response_2d_{eta_sign}_{flav}_{variable}.{'pdf' if PDF else 'png'}",
                        bbox_inches="tight",
                        dpi=300,
                    )

                    plt.close(fig)


if args.histo:
    print("Plotting histograms...")
    eta_pt_bins = []
    for eta in range(len(correct_eta_bins) - 1 if not args.test else 1):
        for pt in range(len(pt_bins) - 1 if not args.test else 1):
            if args.choose_plot:
                index, eta_sign = compute_index_eta(eta)
                if not any(
                    [
                        ((index, eta_sign, pt) == (histo[4], histo[0], histo[5]))
                        for histo in histograms_to_plot
                    ]
                ):
                    continue
            eta_pt_bins.append((eta, pt))
    print("eta_pt_bins", eta_pt_bins)
    with Pool(args.num_processes) as p:
        p.map(functools.partial(plot_histos, histogram_dir=histogram_dir), eta_pt_bins)

if args.no_plot:
    sys.exit()

print("Plotting width...")
with Pool(args.num_processes) as p:
    p.map(
        functools.partial(plot_median_resolution, plot_type="width"),
        range(len(correct_eta_bins) - 1 if not args.test else 1),
    )
print("Plotting inverse medians...")
with Pool(args.num_processes) as p:
    p.map(
        functools.partial(plot_median_resolution, plot_type="inverse_median"),
        # [30],
        range(len(correct_eta_bins) - 1 if not args.test else 1),
    )
# save the fit results
print("Saving fit results...")
write_l2rel_txt(
    main_dir,
    correct_eta_bins,
    year_txt,
    args.num_params,
    VERSION,
    "splitpnetreg15" in args.dir,
    flavs,
)

print("Plotting medians...")
with Pool(args.num_processes) as p:
    p.map(
        functools.partial(plot_median_resolution, plot_type="median"),
        range(len(correct_eta_bins) - 1 if not args.test else 1),
    )


if PLOT_JETPT_MEDIAN:
    print("Plotting average jet pt...")
    with Pool(args.num_processes) as p:
        p.map(
            functools.partial(plot_median_resolution, plot_type="average_jet_pt"),
            range(len(correct_eta_bins) - 1 if not args.test else 1),
        )

print("Plotting resolution...")
with Pool(args.num_processes) as p:
    p.map(
        functools.partial(plot_median_resolution, plot_type="resolution"),
        range(len(correct_eta_bins) - 1 if not args.test else 1),
    )

print("Plotting weighted resolution...")
with Pool(args.num_processes) as p:
    p.map(
        functools.partial(plot_median_resolution, plot_type="weighted_resolution"),
        range(len(correct_eta_bins) - 1 if not args.test else 1),
    )


# print("Plotting 2d median...")
# plot_2d(medians_dict, np.array(pt_bins), np.array(correct_eta_bins))


print("Done!")
