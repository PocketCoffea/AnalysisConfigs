from coffea.util import load
import argparse
import numpy as np
import hist

parser = argparse.ArgumentParser(description="Run the jme analysis")
parser.add_argument(
    "-d",
    "--dir",
    type=str,
    help="Input dir",
)
args = parser.parse_args()

main_dir = args.dir
flavs = ["inclusive", "b", "c", "uds", "g"]

for flav in flavs:

    for eta_sign in ["neg", "pos"]:
        response_to_join = dict()
        eta_cats = ["negneg", "neg"] if eta_sign == "neg" else ["pos", "pospos"]
        for eta_cat in eta_cats:
            response_to_join[eta_cat] = dict()

            print("eta_cat", eta_cat, "flav", flav)

            o = load(f"{main_dir}/{eta_cat}eta_{flav}flav_pnet/output_all.coffea")
            variables = o["variables"].keys()
            for variable in variables:
                histos_dict = o["variables"][variable]
                for sample in histos_dict.keys():
                    for dataset in histos_dict[sample].keys():
                        histo = histos_dict[sample][dataset]
                        response_to_join[eta_cat][variable] = histo
                        # print("histo", histo)

                        # categories = list(histo.axes["cat"])
                        # # print("categories", categories)

                        # # remove the baseline category
                        # (
                        #     categories.remove("baseline")
                        #     if "baseline" in categories
                        #     else None
                        # )

                        # # order the categories so that the ranges in eta are increasing
                        # categories = sorted(
                        #     categories,
                        #     key=lambda x: float(x.split("eta")[1].split("to")[0]),
                        # )
                        # variations = list(histo.axes["variation"])
                        # lenght = len(categories)
                        # for i in range(lenght):

                        #     for var in variations:
                        #         h = histo[{"cat": categories[i]}][{"variation": var}]
                        #         # h is a histo2d and we want to find the median of the distribution along the axis MatchedJets.Response
                        #         # for each bin in the axis MatchedJets.pt
                        #         # so we need to loop over the bins in the axis MatchedJets.pt
                        #         try:
                        #             jet_pt = "MatchedJets.pt"
                        #             pt_axis_histo = h.axes[jet_pt]
                        #         except KeyError:
                        #             jet_pt = "MatchedJetsNeutrino.pt"
                        #             pt_axis_histo = h.axes[jet_pt]

                        #         for j in range(len(pt_axis_histo)):
                        #             # print("\n\n eta", categories[i], "pt", h.axes["MatchedJets.pt"][j])
                        #             # get the histo1d for the bin j in the axis MatchedJets.pt
                        #             h1d = h[{jet_pt: j}]

                        #             # get the values of the histo1d
                        #             values = h1d.values()
                        #             bins = h1d.axes[0].edges
                        #             bins_mid = (bins[1:] + bins[:-1]) / 2

                        #             # HERE: uncomment HERE
                        #             bins_mid = bins_mid[1:]
                        #             values = values[1:]

        # join the histos of type Hist for the same variable and different eta_cat
        for variable in variables:
            hist_0 = response_to_join[eta_cats[0]][variable]
            hist_1 = response_to_join[eta_cats[1]][variable]
            print(response_to_join[eta_cats[0]][variable])
            print(response_to_join[eta_cats[1]][variable])
            # join the Hist hist_0 and hist_1

            h = hist.Hist(
                hist.axis.StrCategory(
                    list(hist_0.axes["cat"]) + list(hist_1.axes["cat"]),
                    name="cat",
                    label="Category",
                ),
                hist.axis.StrCategory(
                    list(hist_0.axes["variation"]), name="variation", label="Variation"
                ),
                hist.axis.Variable(
                    edges=hist_0.axes[2].edges,
                    name=hist_0.axes[2].name,
                    label=hist_0.axes[2].label,
                ),
                hist.axis.Variable(
                    edges=hist_0.axes[3].edges,
                    name=hist_0.axes[3].name,
                    label=hist_0.axes[3].label,
                ),
                # hist.storage.Weight(),
            )
            print("h", h)

            # fill the new histo with the values of hist_0 and hist_1
            for cat in hist_0.axes["cat"]:
                for var in hist_0.axes["variation"]:
                    h.fill(
                        cat=cat,
                        variation=var,
                        MatchedJets_pt=hist_0[{"cat": cat}][{"variation": var}].values(),
                        MatchedJets_Response=hist_0[{"cat": cat}][{"variation": var}].axes[
                            "MatchedJets_Response"
                        ].centers(),
                    )

            raise SystemExit
