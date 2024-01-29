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

# from plot_utils import plot_median_resolution, plot_histos

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
    help="Plot the response histograms",
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

args = parser.parse_args()

localdir = os.path.dirname(os.path.abspath(__file__))

flavs = (
    {
        ("inclusive",): ["."],
        ("b", "c"): [".", "x"],
        ("uds", "g"): [".", "x"],
    }
    if args.full
    else {(args.flav,): ["."]}
)

flavs_not_inclusive = ["", "_b_", "_c_", "_uds_", "_g_"]

variables_colors = {
    "ResponseJEC": "blue",
    "ResponseRaw": "green",
    "ResponsePNetReg": "red",
    # "ResponsePNetRegNeutrino": "orange",
    # "ResponsePNetRegFull": "purple",
}

# set global variables
os.environ["SIGN"] = ""

main_dir = args.dir

if not args.full:
    median_dir = (
        f"{main_dir}/median_plots_unbinned"
        if args.unbinned
        else f"{main_dir}/median_plots_binned"
    )
    os.makedirs(f"{median_dir}", exist_ok=True)
    print("median_dir", median_dir)
    resolution_dir = (
        f"{main_dir}/resolution_plots_unbinned"
        if args.unbinned
        else f"{main_dir}/resolution_plots_binned"
    )
    os.makedirs(f"{resolution_dir}", exist_ok=True)
    print("resolution_dir", resolution_dir)
    if args.histograms:
        response_dir = (
            f"{main_dir}/response_plots_unbinned"
            if args.unbinned
            else f"{main_dir}/response_plots_binned"
        )
        os.makedirs(f"{response_dir}", exist_ok=True)
        print("response_dir", response_dir)

plot_2d_dir = f"{main_dir}/2d_plots"
os.makedirs(f"{plot_2d_dir}", exist_ok=True)
print("plot_2d_dir", plot_2d_dir)


correct_eta_bins = eta_bins

if args.load:
    print("loading response from file")
    medians_dict = dict()
    err_medians_dict = dict()
    resolutions_dict = dict()
    if args.histograms:
        response_dict = dict()
    for eta_sign in ["neg", "pos"] if not args.central else ["central"]:
        medians_dict[eta_sign] = dict()
        err_medians_dict[eta_sign] = dict()
        resolutions_dict[eta_sign] = dict()
        if args.histograms:
            response_dict[eta_sign] = dict()
        for flav_group in flavs:
            medians_dict[eta_sign][flav_group] = dict()
            err_medians_dict[eta_sign][flav_group] = dict()
            resolutions_dict[eta_sign][flav_group] = dict()
            if args.histograms:
                response_dict[eta_sign][flav_group] = dict()
            for flav in flav_group:
                medians_dict[eta_sign][flav_group][flav] = dict()
                err_medians_dict[eta_sign][flav_group][flav] = dict()
                resolutions_dict[eta_sign][flav_group][flav] = dict()
                if args.histograms:
                    response_dict[eta_sign][flav_group][flav] = dict()
                print("eta_sign", eta_sign, "flav_group", flav_group, "flav", flav)
                for variable in variables_colors.keys():
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
                        if args.histograms:
                            response_dir = (
                                f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/response_plots_unbinned"
                                if args.unbinned
                                else f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/response_plots_binned"
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
                    if args.histograms:
                        response_dict[eta_sign][flav_group][flav][variable] = np.load(
                            f"{response_dir}/response_{eta_sign}_{flav}_{variable}.npy"
                        )
    # else:
    #     medians_dict = np.load(f"{median_dir}/medians.npy")
    #     err_medians_dict = np.load(f"{median_dir}/err_medians.npy")
    print("loaded", medians_dict, err_medians_dict)


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
        response_dict = dict()
        # medians = list(list())
        # err_medians = list(list())

        o = load(f"{main_dir}/output_all.coffea") if not args.full else None
        variables = o["variables"].keys() if not args.full else None
        for eta_sign in ["neg", "pos"] if not args.central else ["central"]:
            medians_dict[eta_sign] = dict()
            err_medians_dict[eta_sign] = dict()
            resolutions_dict[eta_sign] = dict()
            response_dict[eta_sign] = dict()

            for flav_group in flavs:
                medians_dict[eta_sign][flav_group] = dict()
                err_medians_dict[eta_sign][flav_group] = dict()
                resolutions_dict[eta_sign][flav_group] = dict()
                response_dict[eta_sign][flav_group] = dict()
                for flav in flav_group:
                    print("eta_sign", eta_sign, "flav_group", flav_group, "flav", flav)
                    medians_dict[eta_sign][flav_group][flav] = dict()
                    err_medians_dict[eta_sign][flav_group][flav] = dict()
                    resolutions_dict[eta_sign][flav_group][flav] = dict()
                    response_dict[eta_sign][flav_group][flav] = dict()
                    if args.full:
                        o = load(
                            f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/output_all.coffea"
                        )
                        variables = o["variables"].keys()
                    for variable in variables:
                        if "Response" not in variable or "VSpt" not in variable:
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
                        err_medians_dict[eta_sign][flav_group][flav][variable] = list(
                            list()
                        )
                        resolutions_dict[eta_sign][flav_group][flav][variable] = list(
                            list()
                        )
                        response_dict[eta_sign][flav_group][flav][variable] = list(
                            list()
                        )

                        for sample in histos_dict.keys():
                            for dataset in histos_dict[sample].keys():
                                # histo = np.array(histos_dict[sample][dataset][cat][0, 0, :])["value"]
                                histo = histos_dict[sample][dataset]
                                # print(histo)
                                categories = list(histo.axes["cat"])
                                # print("categories", categories)

                                # remove the baseline category
                                categories.remove(
                                    "baseline"
                                ) if "baseline" in categories else None

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
                                    response_dict[eta_sign][flav_group][flav][
                                        variable
                                    ].append(list())
                                    for var in variations:
                                        h = histo[{"cat": categories[i]}][
                                            {"variation": var}
                                        ]
                                        # h is a histo2d and we want to find the median of the distribution along the axis MatchedJets.Response
                                        # for each bin in the axis MatchedJets.pt
                                        # so we need to loop over the bins in the axis MatchedJets.pt
                                        jet_pt = "MatchedJets.pt"

                                        for j in range(len(h.axes[jet_pt])):
                                            # print("\n\n eta", categories[i], "pt", h.axes["MatchedJets.pt"][j])
                                            # get the histo1d for the bin j in the axis MatchedJets.pt
                                            h1d = h[{jet_pt: j}]

                                            # get the values of the histo1d
                                            values = h1d.values()
                                            bins = h1d.axes[0].edges
                                            bins_mid = (bins[1:] + bins[:-1]) / 2

                                            # HERE: uncomment HERE
                                            bins_mid = bins_mid[1:]
                                            values = values[1:]

                                            if args.histograms:
                                                #     h_rebin = Hist(hist.axis.Regular(100, 0.0, 2., name=variable))
                                                #     new_num_bins = 100
                                                #     # rebin the histo1d
                                                #     new_bin_width = (bins[-1] - bins[0]) / new_num_bins
                                                #     new_bins = np.arange(bins[0], bins[-1] + new_bin_width, new_bin_width)
                                                #     values, bins = np.histogram(bins_mid, bins=new_bins, weights=values)
                                                rebin_factor = 20
                                                # get index where bins is > 2
                                                index_2 = np.where(bins > 2)[0][0]

                                                rebinned_bins = np.array(
                                                    bins[:index_2][::rebin_factor]
                                                )
                                                rebinned_values = np.add.reduceat(
                                                    values[:index_2],
                                                    range(
                                                        0,
                                                        len(values[:index_2]),
                                                        rebin_factor,
                                                    ),
                                                )
                                                # print("bins", len(bins), "values", len(values))
                                                # print("rebinned_bins", len(rebinned_bins), "rebinned_values", len(rebinned_values))
                                                # h1d_rebinned = np.histogram(bins_mid, bins=rebinned_bins, weights=rebinned_values)

                                                response_dict[eta_sign][flav_group][
                                                    flav
                                                ][variable][i].append(
                                                    (rebinned_values, rebinned_bins)
                                                )
                                            # print("eta_sign", eta_sign, "flav_group", flav_group, "flav", flav, "variable", variable, "eta", categories[i], "pt", h.axes["MatchedJets.pt"][j])
                                            if all([v <= 1 for v in values]):
                                                for k in range(j, len(h.axes[jet_pt])):
                                                    # print("all values are 0")
                                                    medians_dict[eta_sign][flav_group][
                                                        flav
                                                    ][variable][i].append(np.nan)
                                                    err_medians_dict[eta_sign][
                                                        flav_group
                                                    ][flav][variable][i].append(np.nan)
                                                    resolutions_dict[eta_sign][
                                                        flav_group
                                                    ][flav][variable][i].append(np.nan)
                                                break
                                                continue
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
                                            medians_dict[eta_sign][flav_group][flav][
                                                variable
                                            ][i].append(median)

                                            # print("bins_mid", bins_mid)
                                            mean = np.average(bins_mid, weights=values)
                                            # print("mean", mean)
                                            rms = np.sqrt(
                                                np.average(
                                                    (bins_mid - mean) ** 2,
                                                    weights=values,
                                                )
                                            )
                                            # print("rms", rms)
                                            err_median = (
                                                1.253 * rms / np.sqrt(np.sum(values))
                                            )
                                            # print("sum", np.sum(values))
                                            # print("err_median:\n", err_median)
                                            err_medians_dict[eta_sign][flav_group][
                                                flav
                                            ][variable][i].append(err_median)

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
                                            resolutions_dict[eta_sign][flav_group][
                                                flav
                                            ][variable][i].append(resolution)

                        medians_dict[eta_sign][flav_group][flav][variable] = np.array(
                            medians_dict[eta_sign][flav_group][flav][variable]
                        )
                        err_medians_dict[eta_sign][flav_group][flav][
                            variable
                        ] = np.array(
                            err_medians_dict[eta_sign][flav_group][flav][variable]
                        )
                        resolutions_dict[eta_sign][flav_group][flav][
                            variable
                        ] = np.array(
                            resolutions_dict[eta_sign][flav_group][flav][variable]
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

    print("medians", medians_dict)
    print("err_medians", err_medians_dict)
    print("resolution", resolutions_dict)

    for eta_sign in medians_dict.keys():
        for flav_group in medians_dict[eta_sign].keys():
            for flav in medians_dict[eta_sign][flav_group].keys():
                for variable in medians_dict[eta_sign][flav_group][flav].keys():
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
                    np.save(
                        f"{median_dir.replace('/pnfs/psi.ch/cms/trivcat/store/user/mmalucch//out_jme/', '')}/medians_{eta_sign}_{flav}_{variable}.npy",
                        medians_dict[eta_sign][flav_group][flav][variable],
                    )
                    np.save(
                        f"{median_dir.replace('/pnfs/psi.ch/cms/trivcat/store/user/mmalucch//out_jme/', '')}/err_medians_{eta_sign}_{flav}_{variable}.npy",
                        err_medians_dict[eta_sign][flav_group][flav][variable],
                    )
                    np.save(
                        f"{resolution_dir.replace('/pnfs/psi.ch/cms/trivcat/store/user/mmalucch//out_jme/', '')}/resolution_{eta_sign}_{flav}_{variable}.npy",
                        resolutions_dict[eta_sign][flav_group][flav][variable],
                    )
                    if args.histograms:
                        if args.full:
                            response_dir = (
                                f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/response_plots_unbinned"
                                if args.unbinned
                                else f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/response_plots_binned"
                            )
                            os.makedirs(f"{response_dir}", exist_ok=True)
                        np.save(
                            f"{response_dir.replace('/pnfs/psi.ch/cms/trivcat/store/user/mmalucch//out_jme/', '')}/response_{eta_sign}_{flav}_{variable}.npy",
                            response_dict[eta_sign][flav_group][flav][variable],
                        )
if args.central:
    correct_eta_bins = [-5.191, -1.3, 1.3, 5.191]

print("correct_eta_bins", correct_eta_bins, len(correct_eta_bins))


def plot_median_resolution(i, plot_type):
    if not args.central:
        index = int(i % ((len(correct_eta_bins) - 1) / 2))
        if i >= (len(correct_eta_bins) - 1) / 2:
            eta_sign = "pos"
        else:
            eta_sign = "neg"
    else:
        index = i
        eta_sign = "central"

    if plot_type == "median":
        plot_dict = medians_dict
        err_plot_dict = err_medians_dict
    elif plot_type == "resolution":
        plot_dict = resolutions_dict
        err_plot_dict = None
    else:
        print("plot_type not valid")
        return

    # for eta_sign in medians_dict.keys():
    for flav_group in plot_dict[eta_sign].keys():
        # print("plotting median", flav_group, "eta", eta_sign)
        if plot_type == "median":
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
            f"{correct_eta_bins[i]} <" + r"$\eta^{Gen}$" + f"< {correct_eta_bins[i+1]}",
            horizontalalignment="right",
            verticalalignment="top",
            transform=ax.transAxes,
            # bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9),
        )

        j = 0
        plot = False
        max_value = 0
        for flav in plot_dict[eta_sign][flav_group].keys():
            for variable in plot_dict[eta_sign][flav_group][flav].keys():
                plot_array = plot_dict[eta_sign][flav_group][flav][variable]
                max_value = (
                    max(max_value, np.nanmax(plot_array[index, :]))
                    if not np.all(np.isnan(plot_array[index, :]))
                    else max_value
                )
                err_plot_array = (
                    err_plot_dict[eta_sign][flav_group][flav][variable]
                    if err_plot_dict is not None
                    else None
                )
                if variable not in variables_colors.keys():
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
                    pt_bins[1:],
                    plot_array[index, :],
                    yerr=err_plot_array[index, :]
                    if err_plot_array is not None
                    else None,
                    label=f"{variable.replace('Response','')} ({flav.replace('_','') if flav != '' else 'inclusive'})",
                    marker=flavs[flav_group][j],
                    color=variables_colors[variable],
                    linestyle="None",
                )
                if variable == "ResponsePNetReg" and plot_type == "resolution":
                    # plot ratio pnreg / jec
                    jec = plot_dict[eta_sign][flav_group][flav]["ResponseJEC"]
                    gain_res = (jec[index, :] - plot_array[index, :]) / jec[index, :]
                    ax_ratio.errorbar(
                        pt_bins[1:],
                        gain_res,
                        # label= f"{variable.replace('Response','')} / JEC ({flav.replace('_','') if flav != '' else 'inclusive'})",
                        marker=flavs[flav_group][j],
                        color=variables_colors[variable],
                        linestyle="None",
                    )

            j += 1
        # if no variable is plotted, skip
        if plot == False:
            continue
        # check if plot_array is only nan or 0
        if not np.all(np.isnan(plot_array[index, :])) and not np.all(
            plot_array[index, :] == 0
        ):
            ax.set_ylim(top=1.1 * max_value)
        if plot_type == "median":
            ax.set_xlabel(r"$p_{T}^{Gen}$ [GeV]")
        else:
            ax_ratio.set_xlabel(r"$p_{T}^{Gen}$ [GeV]")
        ax.set_ylabel(
            (f"Median (Response)" if args.unbinned else f"Median (Response)")
            if plot_type == "median"
            else (
                f"Resolution (Response)" if args.unbinned else f"Resolution (Response)"
            )
        )
        if plot_type == "resolution":
            ax_ratio.set_ylabel(
                "(JEC - PNetReg) / JEC"
            )  # (r" $\Delta$ resolution / JEC")

        # log x scale
        ax.set_xscale("log")

        ax.legend(frameon=False, ncol=2, loc="upper right")

        ax.grid(color="gray", linestyle="--", linewidth=0.5, which="both")
        if plot_type == "resolution":
            ax_ratio.grid(color="gray", linestyle="--", linewidth=0.5, which="both")
        # hep.style.use("CMS")

        # create string for flavour
        flav_str = ""
        for flav in flav_group:
            flav_str += flav.replace("_", "")

        plots_dir = median_dir if plot_type == "median" else resolution_dir

        if args.full:
            plots_dir = (
                [
                    f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/median_plots_unbinned"
                    if args.unbinned
                    else f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/median_plots_binned"
                    for flav in flav_group
                ]
                if plot_type == "median"
                else [
                    f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/resolution_plots_unbinned"
                    if args.unbinned
                    else f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/resolution_plots_binned"
                    for flav in flav_group
                ]
            )

        # print("ok")
        if type(plots_dir) == str:
            fig.savefig(
                f"{plots_dir.replace('/pnfs/psi.ch/cms/trivcat/store/user/mmalucch//out_jme/', '')}/{'median' if plot_type == 'median' else 'resolution'}_Response_{flav_str}_eta{correct_eta_bins[i]}to{correct_eta_bins[i+1]}.png",
                bbox_inches="tight",
                dpi=300,
            )
        else:
            for plot_dir in plots_dir:
                fig.savefig(
                    f"{plot_dir.replace('/pnfs/psi.ch/cms/trivcat/store/user/mmalucch//out_jme/', '')}/{'median' if plot_type == 'median' else 'resolution'}_Response_{flav_str}_eta{correct_eta_bins[i]}to{correct_eta_bins[i+1]}.png",
                    bbox_inches="tight",
                    dpi=300,
                )
        plt.close(fig)


def plot_histos(eta_pt, response_dir):
    eta_bin = eta_pt[0]
    pt_bin = eta_pt[1]
    if not args.central:
        index = int(eta_bin % ((len(correct_eta_bins) - 1) / 2))
        if eta_bin >= (len(correct_eta_bins) - 1) / 2:
            eta_sign = "pos"
        else:
            eta_sign = "neg"
    else:
        index = eta_bin
        eta_sign = "central"

    # for eta_sign in medians_dict.keys():
    for flav_group in response_dict[eta_sign].keys():
        # print("plotting histos", flav_group, "eta", eta_sign    )
        for flav in response_dict[eta_sign][flav_group].keys():
            fig_tot, ax_tot = plt.subplots()
            max_value = 0
            for variable in response_dict[eta_sign][flav_group][flav].keys():
                histos = response_dict[eta_sign][flav_group][flav][variable]
                if variable not in variables_colors.keys() or index >= len(histos):
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
                fig, ax = plt.subplots()
                # h = histos[index][pt_bin]
                # ax.hist(
                #     h.axes[0].centers,
                #     bins=h.axes[0].edges,
                #     weights=h.values(),
                #     histtype="step",
                #     label=variable,
                #     color=variables_colors[variable],
                # )
                values = histos[index][pt_bin][0]
                bins = histos[index][pt_bin][1]
                max_value = max(max_value, np.nanmax(values))
                # bins_mid = (bins[1:] + bins[:-1]) / 2
                # print("values", len(values), "bins", len(bins), "bins_mid", len(bins_mid))
                ax.hist(
                    bins,
                    bins=bins,
                    weights=values,
                    histtype="step",
                    label=f'{variable.replace("Response", "")} ({flav})',
                    color=variables_colors[variable],
                    density=True,
                )

                ax_tot.hist(
                    bins,
                    bins=bins,
                    weights=values,
                    histtype="step",
                    label=f'{variable.replace("Response", "")} ({flav})',
                    color=variables_colors[variable],
                    density=True,
                )

                # write axis name in latex
                ax.set_xlabel(f"Response")
                ax.set_ylabel(f"Events")
                # if np.any(values != np.nan) and np.any(values != 0):
                #     ax.set_ylim(top=1.3 * np.nanmax(values))

                ax.legend(frameon=False, loc="upper right")

                plt.grid(color="gray", linestyle="--", linewidth=0.5, which="both")
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

                if args.full:
                    response_dir = (
                        f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/response_plots_unbinned"
                        if args.unbinned
                        else f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/response_plots_binned"
                    )
                    os.makedirs(f"{response_dir}", exist_ok=True)

                fig.savefig(
                    f"{response_dir.replace('/pnfs/psi.ch/cms/trivcat/store/user/mmalucch//out_jme/', '')}/histos_{variable}_{flav}_eta{correct_eta_bins[eta_bin]}to{correct_eta_bins[eta_bin+1]}_pt{pt_bins[pt_bin]}to{pt_bins[pt_bin+1]}.png",
                    bbox_inches="tight",
                    dpi=300,
                )
                plt.close(fig)

            # write axis name in latex
            ax_tot.set_xlabel(f"Response")
            ax_tot.set_ylabel(f"Events")

            ax_tot.legend(frameon=False, loc="upper right")
            # if np.any(values != np.nan) and np.any(values != 0):
            #     ax_tot.set_ylim(top=1.3 * max_value)

            plt.grid(color="gray", linestyle="--", linewidth=0.5, which="both")
            # hep.style.use("CMS")
            hep.cms.label(
                year="2022",
                com="13.6",
                label=f"Private Work",
            )
            # write a string on the plot

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
            fig_tot.savefig(
                f"{response_dir.replace('/pnfs/psi.ch/cms/trivcat/store/user/mmalucch//out_jme/', '')}/histos_ResponseAll_{flav}_eta{correct_eta_bins[eta_bin]}to{correct_eta_bins[eta_bin+1]}_pt{pt_bins[pt_bin]}to{pt_bins[pt_bin+1]}.png",
                bbox_inches="tight",
                dpi=300,
            )
            plt.close(fig_tot)


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

    for eta_sign in plot_dict.keys(): #["full"]:
        for flav_group in plot_dict[eta_sign].keys():
            # print("plotting median", flav_group, "eta", eta_sign)

            for flav in plot_dict[eta_sign][flav_group].keys():
                for variable in plot_dict[eta_sign][flav_group][flav].keys():
                    if variable not in variables_colors.keys():
                        continue
                    h_2d = plot_dict[eta_sign][flav_group][flav][variable]

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
                    # print(h_2d)
                    # put zeros instead of nan
                    # h_2d[np.isnan(h_2d)] = 0
                    # print(h_2d)
                    len_eta=int((len(correct_eta_bins_2d)-1)/2)
                    # require pt > 30 and < 1000
                    mask_pt=(pt_bins_2d >= 30) & (pt_bins_2d <= 3000)
                    mask_eta=abs(correct_eta_bins_2d[:len_eta+1]) < 4
                    # print(pt_bins_2d, len(pt_bins_2d))
                    # print(correct_eta_bins_2d, len(correct_eta_bins_2d))
                    # print("h2d", h_2d[-1])
                    pt_bins_2d_cut=pt_bins_2d[mask_pt]
                    correct_eta_bins_2d_cut=correct_eta_bins_2d[:len_eta+1][mask_eta] if eta_sign == "neg" else correct_eta_bins_2d[len_eta:][mask_eta]
                    h_2d=h_2d[mask_eta[:-1],:]
                    # print("h2d0", h_2d[-1], len(h_2d),   len(h_2d[0]))
                    h_2d=h_2d[:, mask_pt[:-1]][:, :-1]
                    # print(pt_bins_2d, len(pt_bins_2d))
                    # print(correct_eta_bins_2d, len(correct_eta_bins_2d))
                    # print("h2d1", h_2d[-1], len(h_2d),   len(h_2d[0]))
                    c=plt.pcolormesh(
                        pt_bins_2d_cut, correct_eta_bins_2d_cut,
                        h_2d,
                        cmap="viridis",
                        # norm=LogNorm(vmin=0.0001, vmax=1),
                        # vmin=0.95,
                        # vmax=1.05,
                        # label=f"{variable.replace('Response','')} ({flav.replace('_','') if flav != '' else 'inclusive'})",
                    )
                    plt.colorbar(c)
                    ax.text(
                        0.98,
                        0.8 if eta_sign == "pos" else 0.2,
                        f"{variable.replace('Response','')} ({flav.replace('_','') if flav != '' else 'inclusive'})",
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
                        f"{plot_2d_dir.replace('/pnfs/psi.ch/cms/trivcat/store/user/mmalucch//out_jme/', '')}/median_response_2d_{eta_sign}_{flav}_{variable}.png",
                        bbox_inches="tight",
                        dpi=300,
                    )

                    plt.close(fig)


print("Plotting 2d median...")
plot_2d(medians_dict, np.array(pt_bins), np.array(correct_eta_bins))

if args.no_plot:
    sys.exit()

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


if args.histograms:
    print("Plotting histograms...")
    eta_pt_bins = []
    for eta in range(len(correct_eta_bins) - 1 if not args.test else 1):
        for pt in range(len(pt_bins) - 1):
            eta_pt_bins.append((eta, pt))
    with Pool(args.num_processes) as p:
        p.map(functools.partial(plot_histos, response_dir=response_dir), eta_pt_bins)

# print(median_dir)
print("Done!")
