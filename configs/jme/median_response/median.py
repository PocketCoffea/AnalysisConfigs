from coffea.util import load
from pocket_coffea.parameters.histograms import *
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
args = parser.parse_args()

localdir = os.path.dirname(os.path.abspath(__file__))

flavs = {
    ("inclusive",): ["."],
    ("b", "c"): [".", "x"],
    ("uds", "g"): [".", "x"],
}
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
    resolution_dir = (
        f"{main_dir}/resolution_plots_unbinned"
        if args.unbinned
        else f"{main_dir}/resolution_plots_binned"
    )
    os.makedirs(f"{resolution_dir}", exist_ok=True)
    if args.histograms:
        response_dir = (
            f"{main_dir}/response_plots_unbinned"
            if args.unbinned
            else f"{main_dir}/response_plots_binned"
        )
        os.makedirs(f"{response_dir}", exist_ok=True)



correct_eta_bins = eta_bins

if args.load:
    # if args.full:
    #     for eta_sign in ["neg", "pos"]:

    # TODO: load the npy files with the correct name for all variables
    medians = np.load(f"{median_dir}/medians.npy")
    err_medians = np.load(f"{median_dir}/err_medians.npy")
    print("loaded", medians, err_medians)


else:
    if args.unbinned:
        print("unbinned")
        medians = np.zeros((len(correct_eta_bins) - 1, len(pt_bins) - 1))
        err_medians = np.zeros((len(correct_eta_bins) - 1, len(pt_bins) - 1))
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
                        medians[i, j] = median
                        # print("median:\n", median)

                        # compute the error on the median as 1.253 * hist->GetRMS() / TMath::Sqrt(hist->GetEffectiveEntries()); but for an unbinned distribution
                        mean = np.mean(column)
                        rms = np.sqrt(np.mean((column - mean) ** 2))
                        err_median = 1.253 * rms / np.sqrt(len(column))
                        err_medians[i, j] = err_median
                        # print("err_median:\n", err_median)
        medians_dict = {"Response": medians}
        err_medians_dict = {"Response": err_medians}
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
                        # TODO: the check of the flavour must be done when opening the file for the new configuration
                        # if flav not in variable or (
                        #     flav == ""
                        #     and any([f in variable for f in flavs_not_inclusive])
                        # ):
                        #     continue
                        if "Response" not in variable or "VSpt" not in variable:
                            continue
                        histos_dict = o["variables"][variable]
                        # remove MatchedJets and VSpt from the name of the variable if present
                        variable = (
                            variable.replace(
                                "MatchedJets_",
                                "",  # if flav == "" else "MatchedJets", ""
                            ).replace(  # TODO: I removed the _ from the MatchedJets_-> need to add it again?
                                "VSpt", ""
                            )
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
                                        #  TODO: I removed the _ from the flavour-> need to add it again
                                        jet_pt = (
                                            # f"MatchedJets_{flav.replace('_','')}.pt"
                                            # if flav != "inclusive"
                                            # else
                                            "MatchedJets.pt"
                                        )


                                        for j in range(len(h.axes[jet_pt])):
                                            # print("\n\n eta", categories[i], "pt", h.axes["MatchedJets.pt"][j])
                                            # get the histo1d for the bin j in the axis MatchedJets.pt
                                            h1d = h[{jet_pt: j}]

                                            # get the values of the histo1d
                                            values = h1d.values()
                                            bins = h1d.axes[0].edges
                                            bins_mid = (bins[1:] + bins[:-1]) / 2
                                            bins_mid = bins_mid[1:]
                                            values = values[1:]

                                            # TODO: rebinning
                                            if args.histograms:
                                                #     h_rebin = Hist(hist.axis.Regular(100, 0.0, 2., name=variable))
                                                #     new_num_bins = 100
                                                #     # rebin the histo1d
                                                #     new_bin_width = (bins[-1] - bins[0]) / new_num_bins
                                                #     new_bins = np.arange(bins[0], bins[-1] + new_bin_width, new_bin_width)
                                                #     values, bins = np.histogram(bins_mid, bins=new_bins, weights=values)
                                                rebin_factor = 20
                                                # get index where bins is > 2
                                                index_2= np.where(bins > 2)[0][0]

                                                rebinned_bins=np.array(bins[:index_2][::rebin_factor])
                                                rebinned_values = np.add.reduceat(values[:index_2], range(0, len(values[:index_2]), rebin_factor))
                                                # print("bins", len(bins), "values", len(values))
                                                # print("rebinned_bins", len(rebinned_bins), "rebinned_values", len(rebinned_values))
                                                # h1d_rebinned = np.histogram(bins_mid, bins=rebinned_bins, weights=rebinned_values)

                                                response_dict[eta_sign][flav_group][
                                                    flav
                                                ][variable][i].append((rebinned_values, rebinned_bins))
                                            # print("eta_sign", eta_sign, "flav_group", flav_group, "flav", flav, "variable", variable, "eta", categories[i], "pt", h.axes["MatchedJets.pt"][j])
                                            if all([v <= 1 for v in values]):
                                                # print("all values are 0")
                                                medians_dict[eta_sign][flav_group][
                                                    flav
                                                ][variable][i].append(np.nan)
                                                err_medians_dict[eta_sign][flav_group][
                                                    flav
                                                ][variable][i].append(np.nan)
                                                resolutions_dict[eta_sign][flav_group][
                                                    flav
                                                ][variable][i].append(np.nan)
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
                                            resolution = (percentile_84 - percentile_16) / 2
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
                        resolutions_dict[eta_sign][flav_group][flav][variable] = np.array(
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
        if args.central:
            correct_eta_bins = [-1.3, 1.3]

        print("correct_eta_bins", correct_eta_bins)

    print("medians", medians_dict)
    print("err_medians", err_medians_dict)
    print("resolution", resolutions_dict)

    for eta_sign in medians_dict.keys():
        # TODO: save the medians in the correct directory
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
                    if args.histograms:
                        response_dir = (
                            f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/response_plots_unbinned"
                            if args.unbinned
                            else f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/response_plots_binned"
                        )
                        os.makedirs(f"{response_dir}", exist_ok=True)
                        np.save(
                            f"{response_dir}/response_{eta_sign}_{flav}_{variable}.npy",
                            response_dict[eta_sign][flav_group][flav][variable],
                        )


def plot_median_resolution(i, plot_type):
    if not args.central:
        index = int(i % ((len(correct_eta_bins) - 1) / 2))
        if i >= (len(correct_eta_bins) - 1) / 2:
            eta_sign = "pos"
        else:
            eta_sign = "neg"
    else:
        index = 0
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
        fig, ax = plt.subplots()
        j = 0
        plot = False
        for flav in plot_dict[eta_sign][flav_group].keys():
            for variable in plot_dict[eta_sign][flav_group][flav].keys():
                plot_array = plot_dict[eta_sign][flav_group][flav][variable]
                err_plot_array = err_plot_dict[eta_sign][flav_group][flav][variable] if err_plot_dict is not None else None
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
                    yerr=err_plot_array[index, :] if err_plot_array is not None else None,
                    label=f"{variable.replace('Response','')} ({flav.replace('_','') if flav != '' else 'inclusive'})",
                    marker=flavs[flav_group][j],
                    color=variables_colors[variable],
                    linestyle="None",
                )
            j += 1
        # if no variable is plotted, skip
        if plot == False:
            continue
        # write axis name in latex
        ax.set_xlabel(r"$p_{T}^{Gen}$ [GeV]")
        ax.set_ylabel(
            (f"Median (Response) unbinned"
            if args.unbinned
            else f"Median (Response) binned") if plot_type == "median" else (
                f"Resolution (Response) unbinned"
                if args.unbinned
                else f"Resolution (Response) binned"
            )
        )
        # log x scale
        ax.set_xscale("log")
        # remove border of legend
        ax.legend(frameon=False, ncol=2)

        plt.grid(color="gray", linestyle="--", linewidth=0.5, which="both")
        # hep.style.use("CMS")
        hep.cms.label(
            year="2022",
            com="13.6",
            label=f"Private Work ({correct_eta_bins[i]} <"
            + r"$\eta^{Gen}$"
            + f"< {correct_eta_bins[i+1]})",
        )

        # create string for flavour
        flav_str = ""
        for flav in flav_group:
            flav_str += flav.replace("_", "")

        plots_dir = median_dir if plot_type == "median" else resolution_dir

        if args.full:
            plots_dir = [
                f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/median_plots_unbinned"
                if args.unbinned
                else f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/median_plots_binned"
                for flav in flav_group
            ] if plot_type == "median" else [
                f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/resolution_plots_unbinned"
                if args.unbinned
                else f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/resolution_plots_binned"
                for flav in flav_group
            ]

        # print("ok")
        if type(plots_dir) == str:
            fig.savefig(
                f"{plots_dir}/{'median' if plot_type == 'median' else 'resolution'}_Response_{flav_str}_eta{correct_eta_bins[i]}to{correct_eta_bins[i+1]}.png",
                bbox_inches="tight",
                dpi=300,
            )
        else:
            for plot_dir in plots_dir:
                fig.savefig(
                    f"{plot_dir}/{'median' if plot_type == 'median' else 'resolution'}_Response_{flav_str}_eta{correct_eta_bins[i]}to{correct_eta_bins[i+1]}.png",
                    bbox_inches="tight",
                    dpi=300,
                )
        plt.close(fig)


def plot_histos(eta_pt):
    eta_bin = eta_pt[0]
    pt_bin = eta_pt[1]
    if not args.central:
        index = int(eta_bin % ((len(correct_eta_bins) - 1) / 2))
        if eta_bin >= (len(correct_eta_bins) - 1) / 2:
            eta_sign = "pos"
        else:
            eta_sign = "neg"
    else:
        index = 0
        eta_sign = "central"

    # for eta_sign in medians_dict.keys():
    for flav_group in response_dict[eta_sign].keys():
        # print("plotting histos", flav_group, "eta", eta_sign    )
        for flav in response_dict[eta_sign][flav_group].keys():
            for variable in response_dict[eta_sign][flav_group][flav].keys():
                histos = response_dict[eta_sign][flav_group][flav][variable]
                if variable not in variables_colors.keys() or index >= len(histos):
                    print("skipping", variable, "index", index, "len(histos)", len(histos))
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
                values= histos[index][pt_bin][0]
                bins= histos[index][pt_bin][1]
                # bins_mid = (bins[1:] + bins[:-1]) / 2
                # print("values", len(values), "bins", len(bins), "bins_mid", len(bins_mid))
                ax.hist(
                    bins,
                    bins=bins,
                    weights=values,
                    histtype="step",
                    label=variable,
                    color=variables_colors[variable],
                )

                # write axis name in latex
                ax.set_xlabel(f"Response")
                ax.set_ylabel(f"Events")
                # remove border of legend
                ax.legend(frameon=False, ncol=2)

                plt.grid(color="gray", linestyle="--", linewidth=0.5, which="both")
                # hep.style.use("CMS")
                hep.cms.label(
                    year="2022",
                    com="13.6",
                    label=f"Private Work",
                )
                # write a string on the plot
                ax.text(
                    0.95,
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
                )


                if args.full:
                    response_dir = (
                        f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/response_plots_unbinned"
                        if args.unbinned
                        else f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/response_plots_binned"
                    )
                    os.makedirs(f"{response_dir}", exist_ok=True)

                fig.savefig(
                    f"{response_dir}/histos_{variable}_{flav}_eta{correct_eta_bins[eta_bin]}to{correct_eta_bins[eta_bin+1]}_pt{pt_bins[pt_bin]}to{pt_bins[pt_bin+1]}.png",
                    bbox_inches="tight",
                    dpi=300,
                )
                plt.close(fig)

print("Plotting medians...")
with Pool(32) as p:
    p.map(functools.partial(plot_median_resolution, plot_type="median"), range(len(correct_eta_bins) - 1 if not args.test else 1))

print("Plotting resolution...")
with Pool(32) as p:
    p.map(functools.partial(plot_median_resolution, plot_type="resolution"), range(len(correct_eta_bins) - 1 if not args.test else 1))


if args.histograms:
    print("Plotting histograms...")
    eta_pt_bins = []
    for eta in range(len(correct_eta_bins) - 1 if not args.test else 1):
        for pt in range(len(pt_bins) - 1):
            eta_pt_bins.append((eta, pt))
    with Pool(32) as p:
        p.map(plot_histos, eta_pt_bins)

# print(median_dir)
print("Done!")
