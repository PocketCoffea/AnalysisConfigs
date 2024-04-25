from coffea.util import load
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import sys
import mplhep as hep
from multiprocessing import Pool
import multiprocessing as mpr
from functools import partial
from hist import Hist
import hist
import functools
import math
import json


import ROOT

ROOT.gROOT.SetBatch(True)

from scipy.optimize import curve_fit
import scipy.stats as stats

sys.path.append("../")
from params.binning import *

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
    "-c",
    "--cartesian",
    action="store_true",
    help="Run cartesian multicuts",
    default=False,
)
parser.add_argument(
    "-histo",
    "--histograms",
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
    "-n",
    "--num-processes",
    type=int,
    help="Number of processes",
    default=32,
)
parser.add_argument(
    "--no-plot",
    action="store_true",
    help="Do not plot",
    default=False,
)
parser.add_argument(
    "--flav",
    help="Flavour",
    type=str,
    default="inclusive",
)
parser.add_argument(
    "--jet-pt",
    help="Consider reco jet pt",
    type=bool,
    default=True,
)

args = parser.parse_args()

localdir = os.path.dirname(os.path.abspath(__file__))

if args.full and args.central:
    flavs = {
        ("inclusive",): ["."],
        ("b", "c"): [".", "x"],
        ("uds", "g"): [".", "x"],
    }
elif args.full:
    flavs = {("inclusive",): ["."]}
else:
    flavs = {(args.flav,): ["."]}


variables_colors = {
    "ResponseJEC": "blue",
    "ResponseRaw": "green",
    "ResponsePNetReg": "red",
    "ResponsePNetRegNeutrino": "purple",
}
if args.jet_pt:
    variables_colors.update(
        {
            "JetPtJEC": "blue",
            "JetPtRaw": "green",
            "JetPtPNetReg": "red",
            "JetPtPNetRegNeutrino": "purple",
        }
    )


# run bash command
os.system("export SIGN=''")

main_dir = args.dir

if not args.full:
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

    if args.histograms:
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
print("plot_2d_dir", plot_2d_dir)


correct_eta_bins = eta_bins

eta_sections = list(eta_sign_dict.keys())

if args.load:
    print("loading from file")
    medians_dict = dict()
    err_medians_dict = dict()
    resolutions_dict = dict()
    if args.histograms:
        histogram_dict = dict()
    for eta_sign in eta_sections if not args.central else ["central"]:
        medians_dict[eta_sign] = dict()
        err_medians_dict[eta_sign] = dict()
        resolutions_dict[eta_sign] = dict()
        if args.histograms:
            histogram_dict[eta_sign] = dict()
        for flav_group in flavs:
            medians_dict[eta_sign][flav_group] = dict()
            err_medians_dict[eta_sign][flav_group] = dict()
            resolutions_dict[eta_sign][flav_group] = dict()
            if args.histograms:
                histogram_dict[eta_sign][flav_group] = dict()
            for flav in flav_group:
                medians_dict[eta_sign][flav_group][flav] = dict()
                err_medians_dict[eta_sign][flav_group][flav] = dict()
                resolutions_dict[eta_sign][flav_group][flav] = dict()
                if args.histograms:
                    histogram_dict[eta_sign][flav_group][flav] = dict()
                print("eta_sign", eta_sign, "flav_group", flav_group, "flav", flav)
                for variable in list(variables_colors.keys()):
                    if args.full:
                        median_dir = (
                            f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/median_plots_unbinned"
                            if args.unbinned
                            else f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/median_plots_binned"
                        )
                        resolution_dir = (
                            f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/resolution_plots_unbinned"
                            if args.unbinned
                            else f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/resolution_plots_binned"
                        )
                        inv_median_dir = (
                            f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/inv_median_plots_unbinned"
                            if args.unbinned
                            else f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/inv_median_plots_binned"
                        )
                        weighted_resolution_dir = (
                            f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/weighted_resolution_plots_unbinned"
                            if args.unbinned
                            else f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/weighted_resolution_plots_binned"
                        )
                        if args.histograms:
                            histogram_dir = (
                                f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/histogram_plots_unbinned"
                                if args.unbinned
                                else f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/histogram_plots_binned"
                            )
                    medians_dict[eta_sign][flav_group][flav][variable] = np.load(
                        f"{median_dir}/medians_{eta_sign}_{flav}_{variable}.npy"
                    )
                    err_medians_dict[eta_sign][flav_group][flav][variable] = np.load(
                        f"{median_dir}/err_medians_{eta_sign}_{flav}_{variable}.npy"
                    )
                    resolutions_dict[eta_sign][flav_group][flav][variable] = np.load(
                        f"{resolution_dir}/resolution_{eta_sign}_{flav}_{variable}.npy"
                    )
                    # TODO: load inverse median and weighted resolution
                    if args.histograms:
                        histogram_dict[eta_sign][flav_group][flav][variable] = np.load(
                            f"{histogram_dir}/histogram_{eta_sign}_{flav}_{variable}.npy",
                            allow_pickle=True,
                        )
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
        histogram_dict = dict()
        # medians = list(list())
        # err_medians = list(list())

        o = load(f"{main_dir}/output_all.coffea") if not args.full else None
        variables = o["variables"].keys() if not args.full else None
        for eta_sign in eta_sections if not args.central else ["central"]:
            medians_dict[eta_sign] = dict()
            err_medians_dict[eta_sign] = dict()
            resolutions_dict[eta_sign] = dict()
            histogram_dict[eta_sign] = dict()

            for flav_group in flavs:
                medians_dict[eta_sign][flav_group] = dict()
                err_medians_dict[eta_sign][flav_group] = dict()
                resolutions_dict[eta_sign][flav_group] = dict()
                histogram_dict[eta_sign][flav_group] = dict()
                for flav in flav_group:
                    print("eta_sign", eta_sign, "flav_group", flav_group, "flav", flav)
                    medians_dict[eta_sign][flav_group][flav] = dict()
                    err_medians_dict[eta_sign][flav_group][flav] = dict()
                    resolutions_dict[eta_sign][flav_group][flav] = dict()
                    histogram_dict[eta_sign][flav_group][flav] = dict()
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
                            histogram_dict[eta_sign][flav_group][flav][variable] = list(
                                list()
                            )

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
                                        histogram_dict[eta_sign][flav_group][flav][
                                            variable
                                        ].append(list())
                                        for var in variations:
                                            h = histo[{"cat": categories[i]}][
                                                {"variation": var}
                                            ]
                                            # h is a histo2d and we want to find the median of the distribution along the axis MatchedJets.Response
                                            # for each bin in the axis MatchedJets.pt
                                            # so we need to loop over the bins in the axis MatchedJets.pt
                                            try:
                                                jet_pt = "MatchedJets.pt"
                                                pt_axis_histo = h.axes[jet_pt]
                                            except KeyError:
                                                jet_pt = "MatchedJetsNeutrino.pt"
                                                pt_axis_histo = h.axes[jet_pt]

                                            for j in range(len(pt_axis_histo)):
                                                # print("\n\n eta", categories[i], "pt", h.axes["MatchedJets.pt"][j])
                                                # get the histo1d for the bin j in the axis MatchedJets.pt
                                                h1d = h[{jet_pt: j}]

                                                # get the values of the histo1d
                                                values = h1d.values()
                                                bins = h1d.axes[0].edges
                                                bins_mid = (bins[1:] + bins[:-1]) / 2

                                                # remove the first bin which is a peak to zero in the response
                                                #HERE
                                                bins_mid = bins_mid[1:]
                                                values = values[1:]
                                                # print("bins_mid", bins_mid)
                                                # print("values", values)

                                                if all([v <= 1 for v in values]):
                                                    for k in range(
                                                        j, len(h.axes[jet_pt])
                                                    ):
                                                        # print("all values are 0")
                                                        medians_dict[eta_sign][
                                                            flav_group
                                                        ][flav][variable][i].append(
                                                            np.nan
                                                        )
                                                        err_medians_dict[eta_sign][
                                                            flav_group
                                                        ][flav][variable][i].append(
                                                            np.nan
                                                        )
                                                        resolutions_dict[eta_sign][
                                                            flav_group
                                                        ][flav][variable][i].append(np.nan)
                                                        histogram_dict[eta_sign][flav_group][
                                                            flav
                                                        ][variable][i].append(
                                                            (np.array([]), np.array([]))
                                                        )
                                                    break
                                                if args.histograms:
                                                    # rebin the histogram
                                                    max_value = (
                                                        2
                                                        if "Response" in variable
                                                        else pt_bins[j + 1] * 1.5
                                                    )
                                                    min_value = (
                                                        1e-3
                                                        if "Response" in variable
                                                        else pt_bins[j] * 0.5
                                                    )
                                                    try:
                                                        max_index = np.where(
                                                            bins > max_value
                                                        )[0][0]
                                                    except IndexError:
                                                        max_index = len(bins) - 1
                                                    min_index = np.where(
                                                        bins < min_value
                                                    )[0][-1]

                                                    rebin_factor = (
                                                        30
                                                        if "Response" in variable
                                                        else max(
                                                            len(
                                                                bins[
                                                                    min_index:max_index
                                                                ]
                                                            )
                                                            // 50,
                                                            1,
                                                        )
                                                    )
                                                    rebinned_bins = np.array(
                                                        bins[min_index:max_index][
                                                            ::rebin_factor
                                                        ]
                                                    )
                                                    rebinned_values = np.add.reduceat(
                                                        values[min_index:max_index],
                                                        range(
                                                            0,
                                                            len(
                                                                values[
                                                                    min_index:max_index
                                                                ]
                                                            ),
                                                            rebin_factor,
                                                        ),
                                                    )
                                                    # if "Response" not in variable:
                                                    #     print("bins", bins[min_index:max_index], "values", values[min_index:max_index])
                                                    #     print("rebinned_bins", rebinned_bins, "rebinned_values", rebinned_values)

                                                    histogram_dict[eta_sign][flav_group][
                                                        flav
                                                    ][variable][i].append(
                                                        (rebinned_values, rebinned_bins)
                                                    )
                                                if "Response" in variable:
                                                    # print("eta_sign", eta_sign, "flav_group", flav_group, "flav", flav, "variable", variable, "eta", categories[i], "pt", h.axes["MatchedJets.pt"][j])
                                                    # print("values", values)
                                                    # print(variable)
                                                    # print("bins_mid", bins_mid)
                                                    # print("values", values)

                                                    # bins_mid = bins_mid[values != 0.]
                                                    # values = values[values != 0.]
                                                    # get the bins of the histo1d
                                                    # find the bin which is the median of the histogram
                                                    cdf = np.cumsum(values)
                                                    # # print("cdf", cdf)
                                                    cdf_normalized = cdf / cdf[-1]
                                                    # # print("cdf_normalized", cdf_normalized)
                                                    median_bin_index = np.argmax(
                                                        cdf_normalized >= 0.5
                                                    )
                                                    # # print("median_bin_index", median_bin_index)
                                                    median = bins_mid[median_bin_index]
                                                    # print("median:\n", median)
                                                    medians_dict[eta_sign][flav_group][
                                                        flav
                                                    ][variable][i].append(median)

                                                    # print("bins_mid", bins_mid)
                                                    mean = np.average(
                                                        bins_mid, weights=values
                                                    )
                                                    # print("mean", mean)
                                                    rms = np.sqrt(
                                                        np.average(
                                                            (bins_mid - mean) ** 2,
                                                            weights=values,
                                                        )
                                                    )
                                                    # print("rms", rms)
                                                    err_median = (
                                                        1.253
                                                        * rms
                                                        / np.sqrt(np.sum(values))
                                                    )
                                                    # print("sum", np.sum(values))
                                                    # print("err_median:\n", err_median)
                                                    err_medians_dict[eta_sign][
                                                        flav_group
                                                    ][flav][variable][i].append(
                                                        err_median
                                                    )

                                                    # define the resolution as the difference between the 84th and 16th percentile
                                                    # find the bin which is the 84th percentile of the histogram
                                                    percentile_84_bin_index = np.argmax(
                                                        cdf_normalized >= 0.84
                                                    )
                                                    # print("percentile_84_bin_index", percentile_84_bin_index)
                                                    percentile_84 = bins_mid[
                                                        percentile_84_bin_index
                                                    ]
                                                    # print("percentile_84", percentile_84)
                                                    # find the bin which is the 16th percentile of the histogram
                                                    percentile_16_bin_index = np.argmax(
                                                        cdf_normalized >= 0.16
                                                    )
                                                    # print("percentile_16_bin_index", percentile_16_bin_index)
                                                    percentile_16 = bins_mid[
                                                        percentile_16_bin_index
                                                    ]
                                                    # print("percentile_16", percentile_16)
                                                    resolution = (
                                                        percentile_84 - percentile_16
                                                    ) / 2
                                                    # print("resolution", resolution)
                                                    resolutions_dict[eta_sign][
                                                        flav_group
                                                    ][flav][variable][i].append(
                                                        resolution
                                                    )
                                                elif "JetPt" in variable:
                                                    # compute the mean of the distribution
                                                    mean = np.average(
                                                        bins_mid, weights=values
                                                    )
                                                    medians_dict[eta_sign][flav_group][
                                                        flav
                                                    ][variable][i].append(mean)
                                                    err_medians_dict[eta_sign][
                                                        flav_group
                                                    ][flav][variable][i].append(np.nan)
                                                    resolutions_dict[eta_sign][
                                                        flav_group
                                                    ][flav][variable][i].append(np.nan)

                            medians_dict[eta_sign][flav_group][flav][variable] = (
                                np.array(
                                    medians_dict[eta_sign][flav_group][flav][variable]
                                )
                            )
                            err_medians_dict[eta_sign][flav_group][flav][variable] = (
                                np.array(
                                    err_medians_dict[eta_sign][flav_group][flav][
                                        variable
                                    ]
                                )
                            )
                            resolutions_dict[eta_sign][flav_group][flav][variable] = (
                                np.array(
                                    resolutions_dict[eta_sign][flav_group][flav][
                                        variable
                                    ]
                                )
                            )
                            # print(
                            #     "medians_dict",
                            #     eta_sign,
                            #     flav_group,
                            #     flav,
                            #     variable,
                            #     medians_dict[eta_sign][flav_group][flav][variable],
                            # )
                            if args.histograms:
                                # for i in range(len(histogram_dict[eta_sign][flav_group][flav][variable])):
                                #     for j in range(len(histogram_dict[eta_sign][flav_group][flav][variable][i])):
                                #         histogram_dict[eta_sign][flav_group][flav][variable][i][j] = (
                                #             np.array(
                                #                 histogram_dict[eta_sign][flav_group][flav][variable][i][j]
                                #             )
                                #         )

                                histogram_dict[eta_sign][flav_group][flav][variable] = (
                                    np.array(
                                        histogram_dict[eta_sign][flav_group][flav][
                                            variable
                                        ]
                                    )
                                )
                                # print(
                                #     "histogram_dict", eta_sign, flav_group, flav, variable,
                                #     histogram_dict[eta_sign][flav_group][flav][variable],
                                # )
                            # sys.exit()

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

    # print("medians", medians_dict)
    # print("err_medians", err_medians_dict)
    # print("resolution", resolutions_dict)

    for eta_sign in medians_dict.keys():
        for flav_group in medians_dict[eta_sign].keys():
            for flav in medians_dict[eta_sign][flav_group].keys():
                if args.full:
                    median_dir = (
                        f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/median_plots_unbinned"
                        if args.unbinned
                        else f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/median_plots_binned"
                    )
                    os.makedirs(f"{median_dir}", exist_ok=True)
                    resolution_dir = (
                        f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/resolution_plots_unbinned"
                        if args.unbinned
                        else f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/resolution_plots_binned"
                    )
                    os.makedirs(f"{resolution_dir}", exist_ok=True)
                    inv_median_dir = (
                        f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/inv_median_plots_unbinned"
                        if args.unbinned
                        else f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/inv_median_plots_binned"
                    )
                    os.makedirs(f"{inv_median_dir}", exist_ok=True)
                    weighted_resolution_dir = (
                        f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/weighted_resolution_plots_unbinned"
                        if args.unbinned
                        else f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/weighted_resolution_plots_binned"
                    )
                    os.makedirs(f"{weighted_resolution_dir}", exist_ok=True)
                    if args.histograms:
                        histogram_dir = (
                            f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/histogram_plots_unbinned"
                            if args.unbinned
                            else f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/histogram_plots_binned"
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
                    # TODO: save inverse median and weighted resolution
                    if args.histograms:
                        # print("eta_sign", eta_sign, "flav_group", flav_group, "flav", flav, "variable", variable)
                        # print(histogram_dict[eta_sign][flav_group][flav][variable])
                        # TODO: save the histograms correctly
                        np.save(
                            f"{histogram_dir}/histogram_{eta_sign}_{flav}_{variable}.npy",
                            histogram_dict[eta_sign][flav_group][flav][variable],
                        )

if args.central:
    correct_eta_bins = central_bins

correct_eta_bins = np.array(correct_eta_bins)
print("correct_eta_bins", correct_eta_bins, len(correct_eta_bins))


def compute_index_eta(eta_bin):

    for eta_sign, eta_interval in eta_sign_dict.items():
        if eta_bin < len(correct_eta_bins[correct_eta_bins < eta_interval[1]]):
            index = eta_bin - len(correct_eta_bins[correct_eta_bins < eta_interval[0]])
            eta_sign = eta_sign
            break

    return index, eta_sign


# define the function to fit with 9 parameters
def std_gaus(x, p0, p1, p2, p3, p4, p5, p6, p7, p8):
    return (
        p0
        + p1 / (np.log10(x) ** 2 + p2)
        + p3 * np.exp(-p4 * (np.log10(x) - p5) ** 2)
        + p6 * np.exp(-p7 * (np.log10(x) - p8) ** 2)
    )


def fit_inv_median(ax, x, y, yerr, variable, y_pos, name_plot):
    print("fit_inv_median", variable)
    p_initial = [
        9.14823123e-01,
        1.59850801e00,
        1.08444406e01,
        -1.65510940e00,
        5.75460089e00,
        -4.40873200e-01,
        -4.04888813e-02,
        1.09142869e02,
        1.43155927e00,
    ]
    p_initial = [
        9.14823123e-01,
        1.59850801e00,
        1.08444406e01,
        -1.65510940e00,
        2.35460089e00,
        1.1,
        -4.04888813e-02,
        1.09142869e02,
        -1.43155927e00,
    ]
    p_initial = [
        1.4079260445453523,
        22.163070215571366,
        32.26551228077457,
        -1.0610367819625621,
        0.016828752007083572,
        0.46245487386104656,
        -4.375311791302624,
        18.346287800110574,
        0.8592087373356424,
    ]
    # do the fit
    popt, pcov = curve_fit(
        std_gaus, x, y, p0=p_initial, sigma=yerr, absolute_sigma=True
    )
    # popt, pcov = p_initial, [[0]*len(p_initial)]*len(p_initial)

    # plot the fit
    x_fit = np.linspace(x[0], x[-1], 1000)
    y_fit = std_gaus(x_fit, *popt)
    ax.plot(x_fit, y_fit, color=variables_colors[variable], linestyle="--")

    # print chi2 and p-value on the plot
    chi2 = np.sum(((y - std_gaus(x, *popt)) / yerr) ** 2)
    ndof = len(x) - len(popt)
    p_value = 1 - stats.chi2.cdf(chi2, ndof)
    ax.text(
        0.98,
        0.2 + y_pos,
        f"$\chi^2$ / ndof = {chi2:.2f} / {ndof}" + f", p-value = {p_value:.2f}",
        horizontalalignment="right",
        verticalalignment="top",
        transform=ax.transAxes,
        color=variables_colors[variable],
        fontsize=8,
    )

    print(
        "\n",
        name_plot,
        "\nx",
        x,
        "\ny",
        y,
        "\nyerr",
        yerr,
        "\npopt",
        popt,
        "\npcov",
        pcov,
        "\nchi2/ndof",
        chi2,
        "/",
        ndof,
        "p_value",
        p_value,
    )
    fit_results = {
        "x": list(x),
        "y": list(y),
        "yerr": list(yerr),
        "parameters": list(popt),
        "errors": list(np.sqrt(np.diag(pcov))),
        "chi2": chi2,
        "ndof": ndof,
        "p_value": p_value,
    }

    return fit_results


def fit_inv_median_root(ax, x, y, yerr, variable, y_pos, name_plot):
    print("fit_inv_median_root", variable)
    # define the function to fit with 9 parameters
    func_string = "[0] + [1] / (TMath::Log10(x) * TMath::Log10(x) + [2]) + [3] * TMath::Exp(-[4] * (TMath::Log10(x) - [5]) * (TMath::Log10(x) - [5])) + [6] * TMath::Exp(-[7] * (TMath::Log10(x) - [8]) * (TMath::Log10(x) - [8]))"
    func_root = ROOT.TF1("func_root", func_string, x[0], x[-1])
    # func_root.SetParameters(
    #     9.14823123e-01,
    #     1.59850801e00,
    #     1.08444406e01,
    #     -1.65510940e00,
    #     2.35460089e00,
    #     1.1,
    #     -4.04888813e-02,
    #     1.09142869e02,
    #     -1.43155927e00,
    # )
    # func_root.SetParameters(
    #     0.9296687827807676,
    #     1.2461504555244276,
    #     7.595369858924967,
    #     -0.49346714415657444,
    #     23.577115065858514,
    #     1.199700717817504,
    #     -0.0404888813,
    #     116.04347276535667,
    #     -1.43155927,
    # )

    # func_root.SetParameter(0,-0.0221278)
    # func_root.SetParameter(1,119.265)
    # func_root.SetParameter(2,100)
    # func_root.SetParameter(3,-0.0679365)
    # func_root.SetParameter(4,2.82597)
    # func_root.SetParameter(5,1.8277)
    # func_root.SetParameter(6,-0.0679365)
    # func_root.SetParameter(7,3.82597)
    # func_root.SetParameter(8,1.8277)
    # func_root.SetParLimits(6,-20,10)
    # func_root.SetParLimits(7,0,100)
    # func_root.SetParLimits(3,-15,15)
    # func_root.SetParLimits(4,0,500)
    # func_root.SetParLimits(0,-2,25)
    # func_root.SetParLimits(1,0,250)

    # fabscor->SetParameter(0,0.0221278);
    # fabscor->SetParLimits(0,-2,50);
    # fabscor->SetParameter(1,14.265);
    # fabscor->SetParLimits(1,0,250);
    # fabscor->SetParameter(2,10);
    # fabscor->SetParLimits(2,0,200);
    # fabscor->SetParameter(3,-0.0679365);
    # fabscor->SetParLimits(3,-15,15);
    # fabscor->SetParameter(4,2.82597);
    # fabscor->SetParLimits(4,0,5);
    # fabscor->SetParameter(5,1.8277);
    # fabscor->SetParLimits(5,0,50);
    # fabscor->SetParameter(6,-0.0679365);
    # fabscor->SetParLimits(6,-20,10);
    # fabscor->SetParameter(7,3.82597);
    # fabscor->SetParLimits(7,0,100);
    # fabscor->SetParameter(8,1.8277);
    # fabscor->SetParLimits(8,-50,50);

    # func_root.SetParameters(
    #     0.0221278,
    #     14.265,
    #     10,
    #     -0.0679365,
    #     2.82597,
    #     1.8277,
    #     -0.0679365,
    #     3.82597,
    #     1.8277,
    # )
    func_root.SetParameters(
        0.4,
        16.265,
        11,
        -0.48,
        5,
        0.8277,
        -0.54,
        0.249482,
        0.9277,
    )
    func_root.SetParLimits(0, -2, 50)
    func_root.SetParLimits(1, 0, 250)
    func_root.SetParLimits(2, 0, 200)
    func_root.SetParLimits(3, -15, 15)
    func_root.SetParLimits(4, 0, 5)
    func_root.SetParLimits(5, 0, 50)
    func_root.SetParLimits(6, -20, 10)
    func_root.SetParLimits(7, 0, 100)
    func_root.SetParLimits(8, -50, 50)

    graph = ROOT.TGraphErrors(len(x), x, y, np.zeros(len(x)), yerr)
    fit_info = str(graph.Fit("func_root", "S N M Q"))  # M E

    param_fit = [func_root.GetParameter(i) for i in range(9)]
    param_err_fit = [func_root.GetParError(i) for i in range(9)]

    # delete graph
    del graph

    # plot the fit
    x_fit = np.linspace(x[0], x[-1], 2000)
    y_fit = std_gaus(x_fit, *param_fit)
    ax.plot(x_fit, y_fit, color=variables_colors[variable], linestyle="--")

    # print chi2 and p-value on the plot
    chi2 = np.sum(((y - std_gaus(x, *param_fit)) / yerr) ** 2)
    ndof = len(x) - len(param_fit)
    p_value = 1 - stats.chi2.cdf(chi2, ndof)
    ax.text(
        0.98,
        0.2 + y_pos,
        f"$\chi^2$ / ndof = {chi2:.2f} / {ndof}" + f", p-value = {p_value:.2f}",
        horizontalalignment="right",
        verticalalignment="top",
        transform=ax.transAxes,
        color=variables_colors[variable],
        fontsize=8,
    )

    if "Invalid FitResult" not in str(fit_info):
        print(
            "\n",
            name_plot,
            "\nx",
            x,
            "\ny",
            y,
            "\nyerr",
            yerr,
            "\npars",
            param_fit,
            "\nerr_pars",
            param_err_fit,
            "\nchi2/ndof",
            chi2,
            "/",
            ndof,
            "p_value",
            p_value,
            "\nfit_info",
            fit_info,
        )

    fit_results = {
        "x": list(x),
        "y": list(y),
        "yerr": list(yerr),
        "parameters": param_fit,
        "errors": param_err_fit,
        "chi2": chi2,
        "ndof": ndof,
        "p_value": p_value,
        "fit_info": fit_info,
    }

    return fit_results


def plot_median_resolution(eta_bin, plot_type):

    if not args.central:
        index, eta_sign = compute_index_eta(eta_bin)
    else:
        index = eta_bin
        eta_sign = "central"

    if "median" in plot_type:
        plot_dict = medians_dict
        err_plot_dict = err_medians_dict
    elif "resolution" in plot_type:
        plot_dict = resolutions_dict
        err_plot_dict = None
    else:
        print("plot_type not valid")
        return

    # for eta_sign in medians_dict.keys():
    for flav_group in plot_dict[eta_sign].keys():
        # create a file to save a dictionary with the fit results
        tot_fit_results = dict()
        # print("plotting median", flav_group, "eta", eta_sign)
        if "median" in plot_type:
            fig, ax = plt.subplots()
        else:
            fig, (ax, ax_ratio) = plt.subplots(
                2, 1, sharex=True, gridspec_kw={"height_ratios": [2.5, 1]}
            )
            fig.tight_layout()
            ax_ratio

        hep.cms.label(
            year="2022",
            com="13.6",
            # label=f"Private Work ({correct_eta_bins[i]} <"
            # + r"$\eta^{Gen}$"
            # + f"< {correct_eta_bins[i+1]})",
            label=f"Private Work",
            ax=ax,
        )

        # write a string on the plot
        ax.text(
            0.98,
            0.7,
            f"{correct_eta_bins[eta_bin]} <"
            + r"$\eta^{Gen}$"
            + f"< {correct_eta_bins[eta_bin+1]}",
            horizontalalignment="right",
            verticalalignment="top",
            transform=ax.transAxes,
            # bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9),
        )

        j = 0
        plot = False
        max_value = 0
        for flav in plot_dict[eta_sign][flav_group].keys():
            y_pos = 0
            for variable in plot_dict[eta_sign][flav_group][flav].keys():
                if "Response" not in variable:
                    continue
                plot_array = plot_dict[eta_sign][flav_group][flav][variable][index, :]
                err_plot_array = (
                    err_plot_dict[eta_sign][flav_group][flav][variable][index, :]
                    if err_plot_dict is not None
                    else None
                )

                if "inverse" in plot_type:
                    err_plot_array = err_plot_array / (plot_array**2)  # * 3  # HERE
                    plot_array = 1 / plot_array
                if "weighted" in plot_type:
                    plot_array = (
                        plot_array
                        * 2
                        / medians_dict[eta_sign][flav_group][flav][variable][index, :]
                    )

                max_value = (
                    max(max_value, np.nanmax(plot_array))
                    if not np.all(np.isnan(plot_array))
                    else max_value
                )

                if (
                    variable not in list(variables_colors.keys())
                    or np.all(
                        np.isnan(
                            plot_dict[eta_sign][flav_group][flav][
                                variable.replace("Response", "JetPt")
                            ][index, :]
                        )
                    )
                    or np.all(np.isnan(plot_array))
                ):
                    continue
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
                plot = True
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
                    label=f"{variable.replace('Response','')} ({flav.replace('_','') if flav != '' else 'inclusive'})",
                    marker=flavs[flav_group][j],
                    color=variables_colors[variable],
                    linestyle="None",
                )
                if (
                    "inverse" in plot_type and "PNet" in variable
                ):  # and "Neutrino" not in variable:
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
                    mask_clip = x > 0  # HERE
                    mask_tot = mask_nan & mask_clip
                    x = x[mask_tot]
                    y = plot_array[mask_tot]
                    y_err = err_plot_array[mask_tot]
                    # print(
                    #     "x",
                    #     x,
                    #     "y",
                    #     y,
                    #     "y_err",
                    #     y_err,
                    #     variable,
                    #     eta_sign,
                    #     flav_group,
                    #     flav,
                    #     index,
                    #     correct_eta_bins[eta_bin],
                    # )
                    if len(x) != 0:
                        fit_results = fit_inv_median_root(
                            ax,
                            x,
                            y,
                            y_err,
                            variable,
                            y_pos,
                            f"{eta_sign} {flav} {correct_eta_bins[eta_bin]} ({index}) {variable}",
                        )
                        y_pos += -0.1
                        tot_fit_results[f"{flav}_{variable}"] = fit_results

                if "ResponsePNetReg" in variable and "resolution" in plot_type:
                    # plot ratio pnreg / jec
                    jec = (
                        plot_dict[eta_sign][flav_group][flav]["ResponseJEC"][index, :]
                        if plot_type == "resolution"
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
                        marker=flavs[flav_group][j],
                        color=variables_colors[variable],
                        linestyle="None",
                    )
                if "median" in plot_type:
                    # plot line at 1 for the whole figure
                    ax.axhline(y=1, color="black", linestyle="-", linewidth=0.7)

            j += 1
        # if no variable is plotted, skip
        if plot == False:
            continue
        # check if plot_array is only nan or 0
        if not np.all(np.isnan(plot_array)) and not np.all(plot_array == 0):
            ax.set_ylim(top=1.1 * max_value)
        if "inverse" in plot_type:
            ax.set_xlabel(r"$p_{T}^{Reco}$ [GeV]")
        elif "median" in plot_type:
            ax.set_xlabel(r"$p_{T}^{Gen}$ [GeV]")
        else:
            ax_ratio.set_xlabel(r"$p_{T}^{Gen}$ [GeV]")

        if plot_type == "median":
            label_y = f"Median (Response)"
        elif plot_type == "inverse_median":
            label_y = r"[Median (Response)]$^{-1}$"
        elif plot_type == "resolution":
            label_y = r"$\frac{q_{84}-q_{16}}{2}$"
        elif plot_type == "weighted_resolution":
            label_y = r"$\frac{q_{84}-q_{16}}{q_{50}}$"

        ax.set_ylabel(label_y)

        # print(
        #     "x",
        #     plot_dict[eta_sign][flav_group][flav][
        #         variable.replace("Response", "JetPt")
        #     ][index, :],
        #     eta_sign,
        #     flav_group,
        #     flav,
        #     index,
        #     correct_eta_bins[eta_bin],
        # )
        # log x scale
        ax.set_xscale("log")

        handles, labels = ax.get_legend_handles_labels()
        handles_dict = dict(zip(labels, handles))
        unique_labels = list(set(handles_dict.keys()))
        unique_dict = {label: handles_dict[label] for label in unique_labels}
        # add elemtns to the legend by hand
        if plot_type == "inverse_median":
            unique_dict["Fit standard+Gaussian"] = plt.Line2D(
                [0], [0], color="black", linestyle="--", label="Fit standard+Gaussian"
            )

        ax.legend(
            unique_dict.values(),
            unique_dict.keys(),
            frameon=False,
            ncol=2,
            loc="upper right",
        )
        # ax.legend(frameon=False, ncol=2, loc="upper right")

        ax.grid(color="gray", linestyle="--", linewidth=0.5, which="both")
        if "resolution" in plot_type:
            ax_ratio.set_ylabel(r"$\Delta$ resolution / JEC")  # "(JEC - PNetReg) / JEC"
            ax_ratio.grid(color="gray", linestyle="--", linewidth=0.5, which="both")
        # hep.style.use("CMS")

        # create string for flavour
        flav_str = ""
        for flav in flav_group:
            flav_str += flav.replace("_", "")

        plots_dir = median_dir if plot_type == "median" else resolution_dir

        if plot_type == "median":
            plots_dir = median_dir
        elif plot_type == "inverse_median":
            plots_dir = inv_median_dir
        elif plot_type == "resolution":
            plots_dir = resolution_dir
        elif plot_type == "weighted_resolution":
            plots_dir = weighted_resolution_dir

        if args.full:
            if plot_type == "median":
                plots_dir = [
                    (
                        f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/median_plots_unbinned"
                        if args.unbinned
                        else f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/median_plots_binned"
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
            elif plot_type == "resolution":
                plots_dir = [
                    (
                        f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/resolution_plots_unbinned"
                        if args.unbinned
                        else f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/resolution_plots_binned"
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
            for plot_dir in plots_dir:
                fig.savefig(
                    f"{plot_dir}/{plot_type}_Response_{flav_str}_eta{correct_eta_bins[eta_bin]}to{correct_eta_bins[eta_bin+1]}.png",
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
                f"{plots_dir}/{plot_type}_Response_{flav_str}_eta{correct_eta_bins[eta_bin]}to{correct_eta_bins[eta_bin+1]}.png",
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
    if not args.central:
        index, eta_sign = compute_index_eta(eta_bin)
    else:
        index = eta_bin
        eta_sign = "central"

    # for eta_sign in medians_dict.keys():
    for flav_group in histogram_dict[eta_sign].keys():
        # print("plotting histos", flav_group, "eta", eta_sign    )
        for flav in histogram_dict[eta_sign][flav_group].keys():
            plot = False
            fig_tot, ax_tot = plt.subplots()
            fig_tot_jetpt, ax_tot_jetpt = plt.subplots()

            max_value = 0
            for variable in histogram_dict[eta_sign][flav_group][flav].keys():
                histos = histogram_dict[eta_sign][flav_group][flav][variable]
                if variable not in list(variables_colors.keys()) or index >= len(
                    histos
                ):
                    print(
                        "skipping", variable, "index", index, "len(histos)", len(histos)
                    )
                    continue
                # for pt_bin in range(len(histos[eta_bin])):
                # print(
                #     "plotting response",
                #     flav_group,
                #     variable,
                #     "eta",
                #     correct_eta_bins[eta_bin],
                #     correct_eta_bins[eta_bin + 1],
                #     "pt",
                #     pt_bins[pt_bin],
                #     pt_bins[pt_bin + 1],
                #     "index",
                #     index,
                # )
                # h = histos[index][pt_bin]
                # ax.hist(
                #     h.axes[0].centers,
                #     bins=h.axes[0].edges,
                #     weights=h.values(),
                #     histtype="step",
                #     label=variable,
                #     color=variables_colors[variable],
                # )
                if args.full:
                    histogram_dir = (
                        f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/histogram_plots_unbinned"
                        if args.unbinned
                        else f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/histogram_plots_binned"
                    )
                    os.makedirs(f"{histogram_dir}", exist_ok=True)
                values = histos[index][pt_bin][0]
                bins = histos[index][pt_bin][1]
                if len(values) == 0:
                    # print(
                    #     "SKIP histo:",
                    #     "flav",
                    #     flav,
                    #     "variable",
                    #     variable,
                    #     "eta",
                    #     correct_eta_bins[eta_bin],
                    #     "pt",
                    #     pt_bins[pt_bin],
                    # )
                    continue
                    break

                plot = True
                fig, ax = plt.subplots()
                max_value = max(max_value, np.nanmax(values))
                # bins_mid = (bins[1:] + bins[:-1]) / 2
                # print("values", len(values), "bins", len(bins), "bins_mid", len(bins_mid))
                ax.hist(
                    bins,
                    bins=bins,
                    weights=values,
                    histtype="step",
                    label=f'{variable.replace("Response", "").replace("JetPt", "")} ({flav})',
                    color=variables_colors[variable],
                    density=True,
                )
                if "Response" in variable:
                    ax_tot.hist(
                        bins,
                        bins=bins,
                        weights=values,
                        histtype="step",
                        label=f'{variable.replace("Response", "")} ({flav})',
                        color=variables_colors[variable],
                        density=True,
                    )
                if "JetPt" in variable:
                    ax_tot_jetpt.hist(
                        bins,
                        bins=bins,
                        weights=values,
                        histtype="step",
                        label=f'{variable.replace("JetPt", "")} ({flav})',
                        color=variables_colors[variable],
                        density=True,
                    )

                # write axis name in latex
                ax.set_xlabel(
                    "Response" if "Response" in variable else r"$p_{T}^{Reco}$"
                )
                ax.set_ylabel(f"Normalized events")
                # if np.any(values != np.nan) and np.any(values != 0):
                #     ax.set_ylim(top=1.3 * np.nanmax(values))

                ax.legend(frameon=False, loc="upper right")

                ax.grid(color="gray", linestyle="--", linewidth=0.5, which="both")
                # hep.style.use("CMS")
                hep.cms.label(
                    year="2022",
                    com="13.6",
                    label=f"Private Work",
                )
                # write a string on the plot
                ax.text(
                    0.98,
                    0.7,
                    f"{correct_eta_bins[eta_bin]} <"
                    + r"$\eta^{Gen}$"
                    + f"< {correct_eta_bins[eta_bin+1]}\n"
                    + f" {int(pt_bins[pt_bin])} <"
                    + r"$p_{T}^{Gen}$"
                    + f"< {int(pt_bins[pt_bin+1])}",
                    horizontalalignment="right",
                    verticalalignment="top",
                    transform=ax.transAxes,
                    # put frame
                    # bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9),
                )

                fig.savefig(
                    f"{histogram_dir}/histos_{variable}_{flav}_eta{correct_eta_bins[eta_bin]}to{correct_eta_bins[eta_bin+1]}_pt{pt_bins[pt_bin]}to{pt_bins[pt_bin+1]}.png",
                    bbox_inches="tight",
                    dpi=300,
                )
                plt.close(fig)

            # check if the plot has plotted histograms
            if not plot:
                plt.close(fig_tot)
                plt.close(fig_tot_jetpt)
                continue

            ax_tot.grid(color="gray", linestyle="--", linewidth=0.5, which="both")
            # write axis name in latex
            ax_tot.set_xlabel(f"Response")
            ax_tot.set_ylabel(f"Normalized events")

            ax_tot_jetpt.set_xlabel(r"$p_{T}^{Reco}$")
            ax_tot_jetpt.set_ylabel(f"Normalized events")

            ax_tot.legend(frameon=False, loc="upper right")
            ax_tot_jetpt.legend(frameon=False, loc="upper right")
            # if np.any(values != np.nan) and np.any(values != 0):
            #     ax_tot.set_ylim(top=1.3 * max_value)

            plt.grid(color="gray", linestyle="--", linewidth=0.5, which="both")
            hep.cms.label(
                year="2022",
                com="13.6",
                label=f"Private Work",
            )

            ax_tot.text(
                0.98,
                0.7,
                f"{correct_eta_bins[eta_bin]} <"
                + r"$\eta^{Gen}$"
                + f"< {correct_eta_bins[eta_bin+1]}\n"
                + f" {int(pt_bins[pt_bin])} <"
                + r"$p_{T}^{Gen}$"
                + f"< {int(pt_bins[pt_bin+1])}",
                horizontalalignment="right",
                verticalalignment="top",
                transform=ax_tot.transAxes,
                # bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9),
            )

            ax_tot_jetpt.text(
                0.98,
                0.7,
                f"{correct_eta_bins[eta_bin]} <"
                + r"$\eta^{Gen}$"
                + f"< {correct_eta_bins[eta_bin+1]}\n"
                + f" {int(pt_bins[pt_bin])} <"
                + r"$p_{T}^{Gen}$"
                + f"< {int(pt_bins[pt_bin+1])}",
                horizontalalignment="right",
                verticalalignment="top",
                transform=ax_tot_jetpt.transAxes,
                # bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9),
            )
            fig_tot.savefig(
                f"{histogram_dir}/histos_ResponseAll_{flav}_eta{correct_eta_bins[eta_bin]}to{correct_eta_bins[eta_bin+1]}_pt{pt_bins[pt_bin]}to{pt_bins[pt_bin+1]}.png",
                bbox_inches="tight",
                dpi=300,
            )
            fig_tot_jetpt.savefig(
                f"{histogram_dir}/histos_JetPtAll_{flav}_eta{correct_eta_bins[eta_bin]}to{correct_eta_bins[eta_bin+1]}_pt{pt_bins[pt_bin]}to{pt_bins[pt_bin+1]}.png",
                bbox_inches="tight",
                dpi=300,
            )
            plt.close(fig_tot)
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
                    if variable not in list(variables_colors.keys()):
                        continue
                    median_2d = plot_dict[eta_sign][flav_group][flav][variable]
                    median_2d = np.array(median_2d)  # -1

                    fig, ax = plt.subplots()
                    hep.cms.label(
                        year="2022",
                        com="13.6",
                        label=f"Private Work",
                        ax=ax,
                    )
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
                        f"Median (response)\n{variable.replace('Response','')} ({flav.replace('_','') if flav != '' else 'inclusive'})",
                        horizontalalignment="right",
                        verticalalignment="top",
                        transform=ax.transAxes,
                        fontsize=10,
                        bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9),
                    )

                    ax.set_xlabel(r"$p_{T}^{Gen}$ [GeV]")
                    ax.set_ylabel(r"$\eta^{Gen}$")
                    ax.grid(color="gray", linestyle="--", linewidth=0.5, which="both")

                    # ax.legend(frameon=False, ncol=2, loc="upper right")

                    fig.savefig(
                        f"{plot_2d_dir}/median_response_2d_{eta_sign}_{flav}_{variable}.png",
                        bbox_inches="tight",
                        dpi=300,
                    )

                    plt.close(fig)


if args.no_plot:
    sys.exit()

print("Plotting inverse medians...")
with Pool(args.num_processes) as p:
    p.map(
        functools.partial(plot_median_resolution, plot_type="inverse_median"),
        # [42],
        range(len(correct_eta_bins) - 1 if not args.test else 1),
    )
# sys.exit()

if args.histograms:
    print("Plotting histograms...")
    eta_pt_bins = []
    for eta in range(len(correct_eta_bins) - 1 if not args.test else 1):
        for pt in range(len(pt_bins) - 1) if not args.test else 1:
            eta_pt_bins.append((eta, pt))
    with Pool(args.num_processes) as p:
        p.map(functools.partial(plot_histos, histogram_dir=histogram_dir), eta_pt_bins)

# print("Plotting 2d median...")
# plot_2d(medians_dict, np.array(pt_bins), np.array(correct_eta_bins))

print("Plotting medians...")
with Pool(args.num_processes) as p:
    p.map(
        functools.partial(plot_median_resolution, plot_type="median"),
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


# print(median_dir)
print("Done!")
