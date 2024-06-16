import awkward as ak
import numpy as np
from matplotlib import pyplot as plt

from response_plot.pol_functions import *
from params.binning import eta_bins


def string_to_pol_function(string):
    # split the string at the + sign
    pol_strings = string.split("+")
    pol = pol_functions_dict[len(pol_strings) - 1]

    return pol


def standard_gaus_partial(x, *params):
    return ((x < params[10]) * (params[9])) + (
        (x >= params[10])
        * (
            params[0]
            + (params[1] / (pow(np.log10(x), 2) + params[2]))
            + (
                params[3]
                * np.exp(
                    -params[4] * ((np.log10(x) - params[5]) * (np.log10(x) - params[5]))
                )
                + (
                    params[6]
                    * np.exp(
                        -params[7]
                        * ((np.log10(x) - params[8]) * (np.log10(x) - params[8]))
                    )
                )
            )
        )
    )


# TODO: implement correctly even for 23BPix
def standard_gaus_function(x, *params):
    # max(0.0001,((x<[10])*([9]))+((x>=[10])*([0]+([1]/(pow(log10(x),2)+[2]))+([3]*exp(-([4]*((log10(x)-[5])*(log10(x)-[5])))))+([6]*exp(-([7]*((log10(x)-[8])*(log10(x)-[8]))))))))

    if type(x) == ak.highlevel.Array:
        return ak.where(
            standard_gaus_partial(x, *params) < 0.0001,
            0.0001,
            standard_gaus_partial(x, *params),
        )
    elif type(x) == float:
        return (
            0.0001
            if standard_gaus_partial(x, *params) < 0.0001
            else standard_gaus_partial(x, *params)
        )
    else:
        raise ValueError("x should be either a float or an awkward array")


def get_closure_function_information(coor_file, use_function=False):
    # open file
    with open(coor_file, "r") as f:
        lines = f.readlines()
        # in the first line the function string is stored after "JetPt" and before "Correction"
        function_string = lines[0].split("JetPt ")[1].split(" Correction")[0].strip()
        phi_bin = True if "JetPhi" in lines[0] else False

        # separate each line in columns
        columns = [line.split() for line in lines[1:]]
        # print(columns[0])
        # get the eta bin edges
        corrections_eta_bins = [
            [float(column[0]) for column in columns],
            [float(column[1]) for column in columns],
        ]
        k=0
        corrections_phi_bins = []
        if phi_bin:
            corrections_phi_bins = [
                [float(column[2]) for column in columns],
                [float(column[3]) for column in columns],
            ]
            k=2


        # get the number of parameters
        num_params = [int(column[k+2]) for column in columns]
        # get the jet pt range
        jet_pt = [
            [float(column[k+3]) for column in columns],
            [float(column[k+4]) for column in columns],
        ]
        # get the parameters
        # params = [[float(column[k+5+i]) for column in columns] for i in range(max(num_params)-2)]
        params = [
            [float(column[k+5 + i]) for i in range(num_params[j] - 2)]
            for j, column in enumerate(columns)
        ]

        # print(corrections_eta_bins[0])
        # print(corrections_eta_bins[1])
        # print(num_params)
        # print(jet_pt[0])
        # print(jet_pt[1])
        # print(params)

        # consier only eta bins in the eta_bins range
        # correct_indeces=[]
        # mask_eta_bins = (corrections_eta_bins[0][0] >= eta_bins[0]) & (corrections_eta_bins[1][0] <= eta_bins[-1])
        # for i in range(len(corrections_eta_bins[0])):
        #     for j in range(len(eta_bins)-1):
        #         if corrections_eta_bins[0][i] >= eta_bins[j] and corrections_eta_bins[1][i] <= eta_bins[j+1]:
        #             correct_indeces.append(i)
        #             break
        # print(correct_indeces)
        # # keep only the correct indeces
        # corrections_eta_bins=[
        #     [corrections_eta_bins[0][i] for i in correct_indeces],
        #     [corrections_eta_bins[1][i] for i in correct_indeces]
        # ]
        # num_params=[num_params[i] for i in correct_indeces]
        # jet_pt=[
        #     [jet_pt[0][i] for i in correct_indeces],
        #     [jet_pt[1][i] for i in correct_indeces]
        # ]

        eta_bins_array = np.array(corrections_eta_bins)
        mask_eta_bins = (eta_bins_array[0] >= eta_bins[0]) & (
            eta_bins_array[1] <= eta_bins[-1]
        )
        corrections_eta_bins = [
            [
                corrections_eta_bins[0][i]
                for i in range(len(corrections_eta_bins[0]))
                if mask_eta_bins[i]
            ],
            [
                corrections_eta_bins[1][i]
                for i in range(len(corrections_eta_bins[1]))
                if mask_eta_bins[i]
            ],
        ]
        if phi_bin:
            corrections_phi_bins = [
                [
                    corrections_phi_bins[0][i]
                    for i in range(len(corrections_phi_bins[0]))
                    if mask_eta_bins[i]
                ],
                [
                    corrections_phi_bins[1][i]
                    for i in range(len(corrections_phi_bins[1]))
                    if mask_eta_bins[i]
                ],
            ]
        num_params = [num_params[i] for i in range(len(num_params)) if mask_eta_bins[i]]
        jet_pt = [
            [jet_pt[0][i] for i in range(len(jet_pt[0])) if mask_eta_bins[i]],
            [jet_pt[1][i] for i in range(len(jet_pt[1])) if mask_eta_bins[i]],
        ]
        params = [
            [params[i][j] for j in range(len(params[i])) if mask_eta_bins[i]]
            for i in range(len(params))
        ]
        # remove empty lists
        params = [i for i in params if i]

        # print("\n\nAfter")
        # print(mask_eta_bins)
        # print(corrections_eta_bins[0])
        # print(corrections_eta_bins[1])
        # print(num_params[0])
        # print(jet_pt[0])
        # print(jet_pt[1])
        # print(params)

        function_dict = {
            "function_string": function_string,
            "corrections_eta_bins": corrections_eta_bins,
            "corrections_phi_bins": corrections_phi_bins,
            "num_params": num_params,
            "jet_pt": jet_pt,
            "params": params,
        }

        if not use_function:
            return function_dict

        # convert the string to one of the pol functions
        function = string_to_pol_function(function_string)

        def def_closure_function(eta, pt):
            # find the right bin
            for i in range(len(corrections_eta_bins[0])):
                if corrections_eta_bins[0][i] <= eta < corrections_eta_bins[1][i]:
                    if jet_pt[0][i] <= pt < jet_pt[1][i]:
                        # calculate the correction
                        return function(pt, *params[i])
                    elif pt < jet_pt[0][i]:
                        return function(jet_pt[0][i], *params[i])
                    else:
                        return function(jet_pt[1][i], *params[i])
            return 1

        def def_closure_function_awkard(eta, pt):
            # eta and pt are awkward arrays
            # find the right bin
            corr = ak.ones_like(eta)
            print(type(pt))
            print(pt.type)
            pt = ak.values_astype(pt, "float32")
            print(pt.type)
            for i in range(len(corrections_eta_bins[0])):
                mask_eta = (corrections_eta_bins[0][i] <= eta) & (
                    eta < corrections_eta_bins[1][i]
                )
                # print(mask_eta)
                mask_pt = (jet_pt[0][i] <= pt) & (pt < jet_pt[1][i])
                # print(mask_pt)
                corr = ak.where(
                    mask_eta,
                    ak.where(
                        mask_pt,
                        function(pt, *params[i]),
                        ak.where(
                            pt < jet_pt[0][i],
                            function(jet_pt[0][i], *params[i]),
                            function(jet_pt[1][i], *params[i]),
                        ),
                    ),
                    corr,
                )
                # print(corr)
                # for j in range(len(corr)):
                #     for k in range(len(corr[j])):
                #         print(corrections_eta_bins[0][i], eta[j][k], pt[j][k], corr[j][k], function(pt, *params[i])[j][k], function(pt[j][k], *params[i]))
            print(type(corr))
            print(corr.type)
            return corr

        return def_closure_function_awkard


if __name__ == "__main__":
    test_closure_function = get_closure_function_information(
        "params/Summer23Run3_PNETREG_MC_L2Relative_AK4PUPPI.txt", use_function=True
    )

    # print(test_closure_function(0.5, 15.19084472))
    # print(test_closure_function(0.5, 15.1))
    # print(test_closure_function(0.5, 4587.0))
    # print(test_closure_function(0.5, 4587.10368490))
    # print(test_closure_function(0.5, 500000))
    # print(test_closure_function(6, 15.1))
    # print(test_closure_function(-5, 20))

    a = ak.highlevel.Array([[-0.8, -0.8], [-0.8, -0.8, -0.8]])
    b = ak.highlevel.Array([[15.1, 2091.427001953125], [15.1, 2000, 5000]])

    # num = 10000
    # a = ak.Array([-5] * num)
    # b = ak.Array(list(np.linspace(10, 5000, num)))

    out = test_closure_function(a, b)

    # print(out)

    for i in range(len(out)):
        for j in range(len(out[i])):
            print(a[i][j], b[i][j], out[i][j])

            print((b**2)[i][j], b[i][j] ** 2)

    # fig, ax = plt.subplots()
    # ax.plot(b, out, label="Closure function")
    # # log scale
    # ax.set_xscale("log")
    # fig.savefig("closure_function.png")
