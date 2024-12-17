from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.cut_functions import (
    get_HLTsel,
)
from pocket_coffea.parameters.histograms import *
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.lib.columns_manager import ColOut
from pocket_coffea.parameters import defaults

from workflow import VBFHH4bbQuarkMatchingProcessor
from custom_cut_functions import *
from custom_cuts import *

import os

localdir = os.path.dirname(os.path.abspath(__file__))

# Loading default parameters

default_parameters = defaults.get_default_parameters()
defaults.register_configuration_dir("config_dir", localdir + "/params")

# adding object preselection
year = "2022_postEE"
parameters = defaults.merge_parameters_from_files(
    default_parameters,
    f"{localdir}/params/object_preselection.yaml",
    f"{localdir}/params/triggers.yaml",
    f"{localdir}/params/jets_calibration.yaml",
    # f"{localdir}/params/plotting_style.yaml",
    update=True,
)

SPANET_MODEL = (
    "params/out_hh4b_5jets_ATLAS_ptreg_c0_lr1e4_wp0_noklininp_oc_300e_kl3p5.onnx"
)
DNN_MODEL="/t3home/rcereghetti/ML_pytorch/out/20241212_223142_SemitTightPtLearningRateConstant/models/model_28.onnx"

HIGGS_PARTON_MATCHING=False
VBF_PARTON_MATCHING = False


# TODO: spostare queste funzioni?

jet_info = ["index", "pt", "btagPNetQvG", "eta", "btagPNetB", "phi", "mass"]


# Combine jet_hists from position start to position end
def jet_hists_dict(coll="JetGood", start=1, end=5):
    combined_dict = {}
    for pos in range(start, end + 1):
        combined_dict.update(
            jet_hists(coll=coll, pos=pos)
        )
    return combined_dict


# Helper function to create HistConf() for a specific configuration
def create_HistConf(coll, field, pos=None, bins=60, start=0, stop=1, label=None):
    axis_params = {
        "coll": coll,
        "field": field,
        "bins": bins,
        "start": start,
        "stop": stop,
        "label": label if label else field,
    }
    if pos is not None:
        axis_params["pos"] = pos
    return {label: HistConf([Axis(**axis_params)])}


cfg = Configurator(
    parameters=parameters,
    datasets={
        "jsons": [
            f"{localdir}/datasets/QCD.json",
            f"{localdir}/datasets/signal_VBF_HH4b_local.json",
            f"{localdir}/datasets/signal_ggF_HH4b_local.json",
        ],
        "filter": {
            "samples": (
                [
                    "VBF_HHto4B",
                    "GluGlutoHHto4B",
                    # TODO qcd
                ]
            ),
            "samples_exclude": [],
            "year": [year],
        },
        "subsamples": {},
    },
    workflow=VBFHH4bbQuarkMatchingProcessor,
    workflow_options={
        "parton_jet_min_dR": 0.4,
        "max_num_jets": 5,
        "which_bquark": "last",
        "spanet_model": SPANET_MODEL if not HIGGS_PARTON_MATCHING else None,
        "DNN_model": DNN_MODEL,
        "vbf_parton_matching": VBF_PARTON_MATCHING,
    },
    skim=[
        get_HLTsel(primaryDatasets=["JetMET"]),
    ],
    preselections=[vbf_hh4b_presel],
    categories={
        **{"4b_region": [hh4b_4b_region]},
        # **{"4b_VBFtight_region": [hh4b_4b_region, VBFtight_region]},
        # **{"4b_VBFtight_region": [hh4b_4b_region, vbf_wrapper()]},
        #
        # **{
        #     f"4b_VBFtight_{list(ab[0].keys())[i]}_region": [
        #         hh4b_4b_region,
        #         vbf_wrapper(ab[i]),
        #     ]
        #     for i in range(0, 6)
        # },
        #
        # **{"4b_VBF_generalSelection_region": [hh4b_4b_region, VBF_generalSelection_region]},
        # **{"4b_VBF_region": [hh4b_4b_region, VBF_region]},
        # **{f"4b_VBF_0{i}qvg_region": [hh4b_4b_region, VBF_region, qvg_regions[f"qvg_0{i}_region"]] for i in range(5, 10)},
        # **{f"4b_VBF_0{i}qvg_generalSelection_region": [hh4b_4b_region, VBF_generalSelection_region, qvg_regions[f"qvg_0{i}_region"]] for i in range(5, 10)},
        # "2b_region": [hh4b_2b_region],
        # TODO
    },
    weights={
        "common": {
            "inclusive": [
                "genWeight",
                "lumi",
                "XS",
            ],
            "bycategory": {},
        },
        "bysample": {},
    },
    variations={
        "weights": {
            "common": {
                "inclusive": [],
                "bycategory": {},
            },
            "bysample": {},
        }
    },
    variables={
        # **count_hist(coll="JetGood", bins=10, start=0, stop=10),
        # **jet_hists_dict(coll="JetGood", start=1, end=5),
        # **create_HistConf("JetGoodVBF", "eta", bins=60, start=-5, stop=5, label="JetGoodVBFeta"),
        # **create_HistConf("JetGoodVBF", "btagPNetQvG", pos=0, bins=60, start=0, stop=1, label="JetGoodVBFQvG_0"),
        # **create_HistConf("JetGoodVBF", "btagPNetQvG", pos=1, bins=60, start=0, stop=1, label="JetGoodVBFQvG_1"),
        # **create_HistConf("events", "deltaEta", bins=60, start=5, stop=10, label="JetGoodVBFdeltaEta"),
        # **create_HistConf("JetVBF_generalSelection", "eta", bins=60, start=-5, stop=5, label="JetVBFgeneralSelectionEta"),
        # **create_HistConf("JetVBF_generalSelection", "btagPNetQvG", pos=0, bins=60, start=0, stop=1, label="JetVBFgeneralSelectionQvG_0"),
        # **create_HistConf("JetVBF_generalSelection", "btagPNetQvG", pos=1, bins=60, start=0, stop=1, label="JetVBFgeneralSelectionQvG_1"),
        #
        #
        # **create_HistConf(
        #     "JetVBF_matched",
        #     "eta",
        #     bins=60,
        #     start=-5,
        #     stop=5,
        #     label="JetVBF_matched_eta",
        # ),
        # **create_HistConf(
        #     "events",
        #     "etaProduct",
        #     bins=5,
        #     start=-2.5,
        #     stop=2.5,
        #     label="JetVBF_matched_eta_product",
        # ),
        # **create_HistConf(
        #     "JetVBF_matched",
        #     "pt",
        #     bins=100,
        #     start=0,
        #     stop=1000,
        #     label="JetVBF_matched_pt",
        # ),
        # **create_HistConf(
        #     "JetVBF_matched",
        #     "btagPNetQvG",
        #     pos=0,
        #     bins=60,
        #     start=0,
        #     stop=1,
        #     label="JetVBF_matchedQvG_0",
        # ),
        # **create_HistConf(
        #     "JetVBF_matched",
        #     "btagPNetQvG",
        #     pos=1,
        #     bins=60,
        #     start=0,
        #     stop=1,
        #     label="JetVBF_matchedQvG_1",
        # ),
        # **create_HistConf(
        #     "quarkVBF_matched",
        #     "eta",
        #     bins=60,
        #     start=-5,
        #     stop=5,
        #     label="quarkVBF_matched_Eta",
        # ),
        # **create_HistConf(
        #     "quarkVBF_matched",
        #     "pt",
        #     bins=100,
        #     start=0,
        #     stop=1000,
        #     label="quarkVBF_matched_pt",
        # ),
        # **create_HistConf(
        #     "JetVBF_matched",
        #     "btagPNetB",
        #     bins=100,
        #     start=0,
        #     stop=1,
        #     label="JetGoodVBF_matched_btag",
        # ),
        # **create_HistConf(
        #     "events", "deltaEta_matched", bins=100, start=0, stop=10, label="deltaEta"
        # ),
        # **create_HistConf(
        #     "events", "jj_mass_matched", bins=100, start=0, stop=5000, label="jj_mass"
        # ),
        # **create_HistConf("HH", "mass", bins=100, start=0, stop=2500, label="HH_mass"),
        **create_HistConf("events", "HH_deltaR", bins=50, start=0, stop=8, label="HH_deltaR"),
        **create_HistConf("events", "H1j1_deltaR", bins=50, start=0, stop=8, label="H1j1_deltaR"),
        **create_HistConf("events", "H1j2_deltaR", bins=50, start=0, stop=8, label="H1j2_deltaR"),
        **create_HistConf("events", "H2j1_deltaR", bins=50, start=0, stop=8, label="H2j1_deltaR"),
        **create_HistConf("events", "HH_centrality", bins=50, start=0, stop=1, label="HH_centrality"),

        **create_HistConf("HH", "pt", bins=100, start=0, stop=800, label="HH_pt"),
        **create_HistConf("HH", "eta", bins=60, start=-6, stop=6, label="HH_eta"),
        **create_HistConf("HH", "phi", bins=60, start=-5, stop=5, label="HH_phi"),
        **create_HistConf("HH", "mass", bins=100, start=0, stop=2200, label="HH_mass"),

        **create_HistConf("HiggsLeading", "pt", bins=100, start=0, stop=800, label="HiggsLeading_pt"),
        **create_HistConf("HiggsLeading", "eta", bins=60, start=-5, stop=5, label="HiggsLeading_eta"),
        **create_HistConf("HiggsLeading", "phi", bins=60, start=-5, stop=5, label="HiggsLeading_phi"),
        **create_HistConf("HiggsLeading", "mass", bins=100, start=0, stop=500, label="HiggsLeading_mass"),

        **create_HistConf("HiggsSubLeading", "pt", bins=100, start=0, stop=800, label="HiggsSubLeading_pt"),
        **create_HistConf("HiggsSubLeading", "eta", bins=60, start=-5, stop=5, label="HiggsSubLeading_eta"),
        **create_HistConf("HiggsSubLeading", "phi", bins=60, start=-5, stop=5, label="HiggsSubLeading_phi"),
        **create_HistConf("HiggsSubLeading", "mass", bins=100, start=0, stop=500, label="HiggsSubLeading_mass"),

        **create_HistConf("Jet", "pt", bins=100, pos=0, start=0, stop=800, label="Jet_pt0"),
        **create_HistConf("Jet", "pt", bins=100, pos=1, start=0, stop=800, label="Jet_pt1"),
        **create_HistConf("Jet", "eta", bins=60, pos=0, start=-5, stop=5, label="Jet_eta0"),
        **create_HistConf("Jet", "eta", bins=60, pos=1, start=-5, stop=5, label="Jet_eta1"),
        **create_HistConf("Jet", "phi", bins=60, pos=0, start=-5, stop=5, label="Jet_phi0"),
        **create_HistConf("Jet", "phi", bins=60, pos=1, start=-5, stop=5, label="Jet_phi1"),
        **create_HistConf("Jet", "mass", bins=100, pos=0, start=0, stop=150, label="Jet_mass0"),
        **create_HistConf("Jet", "mass", bins=100, pos=1, start=0, stop=150, label="Jet_mass1"),
        **create_HistConf("Jet", "btagPNetB", pos=0, bins=100, start=0, stop=1, label="Jet_btagPNetB0"),
        **create_HistConf("Jet", "btagPNetB", pos=1, bins=100, start=0, stop=1, label="Jet_btagPNetB1"),
        **create_HistConf("Jet", "btagPNetQvG", pos=0, bins=100, start=0, stop=1, label="Jet_btagPNetQvG0"),
        **create_HistConf("Jet", "btagPNetQvG", pos=1, bins=100, start=0, stop=1, label="Jet_btagPNetQvG1"),

        **create_HistConf("JetGoodFromHiggsOrdered", "pt", bins=100, pos=0, start=0, stop=700, label="JetGoodFromHiggsOrdered_pt0"),
        **create_HistConf("JetGoodFromHiggsOrdered", "eta", bins=60, pos=0, start=-5, stop=5, label="JetGoodFromHiggsOrdered_eta0"),
        **create_HistConf("JetGoodFromHiggsOrdered", "phi", bins=60, pos=0, start=-5, stop=5, label="JetGoodFromHiggsOrdered_phi0"),
        **create_HistConf("JetGoodFromHiggsOrdered", "mass", bins=100, pos=0, start=0, stop=80, label="JetGoodFromHiggsOrdered_mass0"),
        **create_HistConf("JetGoodFromHiggsOrdered", "btagPNetB", bins=100, pos=0, start=0, stop=1, label="JetGoodFromHiggsOrdered_btagPNetB0"),
        **create_HistConf("JetGoodFromHiggsOrdered", "btagPNetQvG", bins=100, pos=0, start=0, stop=1, label="JetGoodFromHiggsOrdered_btagPNetQvG0"),
        **create_HistConf("JetGoodFromHiggsOrdered", "pt", bins=100, pos=1, start=0, stop=700, label="JetGoodFromHiggsOrdered_pt1"),
        **create_HistConf("JetGoodFromHiggsOrdered", "eta", bins=60, pos=1, start=-5, stop=5, label="JetGoodFromHiggsOrdered_eta1"),
        **create_HistConf("JetGoodFromHiggsOrdered", "phi", bins=60, pos=1, start=-5, stop=5, label="JetGoodFromHiggsOrdered_phi1"),
        **create_HistConf("JetGoodFromHiggsOrdered", "mass", bins=100, pos=1, start=0, stop=80, label="JetGoodFromHiggsOrdered_mass1"),
        **create_HistConf("JetGoodFromHiggsOrdered", "btagPNetB", bins=100, pos=1, start=0, stop=1, label="JetGoodFromHiggsOrdered_btagPNetB1"),
        **create_HistConf("JetGoodFromHiggsOrdered", "btagPNetQvG", bins=100, pos=1, start=0, stop=1, label="JetGoodFromHiggsOrdered_btagPNetQvG1"),
        **create_HistConf("JetGoodFromHiggsOrdered", "pt", bins=100, pos=2, start=0, stop=700, label="JetGoodFromHiggsOrdered_pt2"),
        **create_HistConf("JetGoodFromHiggsOrdered", "eta", bins=60, pos=2, start=-5, stop=5, label="JetGoodFromHiggsOrdered_eta2"),
        **create_HistConf("JetGoodFromHiggsOrdered", "phi", bins=60, pos=2, start=-5, stop=5, label="JetGoodFromHiggsOrdered_phi2"),
        **create_HistConf("JetGoodFromHiggsOrdered", "mass", bins=100, pos=2, start=0, stop=80, label="JetGoodFromHiggsOrdered_mass2"),
        **create_HistConf("JetGoodFromHiggsOrdered", "btagPNetB", bins=100, pos=2, start=0, stop=1, label="JetGoodFromHiggsOrdered_btagPNetB2"),
        **create_HistConf("JetGoodFromHiggsOrdered", "btagPNetQvG", bins=100, pos=2, start=0, stop=1, label="JetGoodFromHiggsOrdered_btagPNetQvG2"),
        **create_HistConf("JetGoodFromHiggsOrdered", "pt", bins=100, pos=3, start=0, stop=700, label="JetGoodFromHiggsOrdered_pt3"),
        **create_HistConf("JetGoodFromHiggsOrdered", "eta", bins=60, pos=3, start=-5, stop=5, label="JetGoodFromHiggsOrdered_eta3"),
        **create_HistConf("JetGoodFromHiggsOrdered", "phi", bins=60, pos=3, start=-5, stop=5, label="JetGoodFromHiggsOrdered_phi3"),
        **create_HistConf("JetGoodFromHiggsOrdered", "mass", bins=100, pos=3, start=0, stop=80, label="JetGoodFromHiggsOrdered_mass3"),
        **create_HistConf("JetGoodFromHiggsOrdered", "btagPNetB", bins=100, pos=3, start=0, stop=1, label="JetGoodFromHiggsOrdered_btagPNetB3"),
        **create_HistConf("JetGoodFromHiggsOrdered", "btagPNetQvG", bins=100, pos=3, start=0, stop=1, label="JetGoodFromHiggsOrdered_btagPNetQvG3"),

        **create_HistConf("JetVBFLeadingPtNotFromHiggs", "pt", bins=100, pos=0, start=0, stop=700, label="JetVBFLeadingPtNotFromHiggs_pt0"),
        **create_HistConf("JetVBFLeadingPtNotFromHiggs", "eta", bins=60, pos=0, start=-5, stop=5, label="JetVBFLeadingPtNotFromHiggs_eta0"),
        **create_HistConf("JetVBFLeadingPtNotFromHiggs", "phi", bins=60, pos=0, start=-5, stop=5, label="JetVBFLeadingPtNotFromHiggs_phi0"),
        **create_HistConf("JetVBFLeadingPtNotFromHiggs", "mass", bins=100, pos=0, start=0, stop=75, label="JetVBFLeadingPtNotFromHiggs_mass0"),
        **create_HistConf("JetVBFLeadingPtNotFromHiggs", "btagPNetB", bins=100, pos=0, start=0, stop=1, label="JetVBFLeadingPtNotFromHiggs_btagPNetB0"),
        **create_HistConf("JetVBFLeadingPtNotFromHiggs", "btagPNetQvG", bins=100, pos=0, start=0, stop=1, label="JetVBFLeadingPtNotFromHiggs_btagPNetQvG0"),
        **create_HistConf("JetVBFLeadingPtNotFromHiggs", "pt", bins=100, pos=1, start=0, stop=700, label="JetVBFLeadingPtNotFromHiggs_pt1"),
        **create_HistConf("JetVBFLeadingPtNotFromHiggs", "eta", bins=60, pos=1, start=-5, stop=5, label="JetVBFLeadingPtNotFromHiggs_eta1"),
        **create_HistConf("JetVBFLeadingPtNotFromHiggs", "phi", bins=60, pos=1, start=-5, stop=5, label="JetVBFLeadingPtNotFromHiggs_phi1"),
        **create_HistConf("JetVBFLeadingPtNotFromHiggs", "mass", bins=100, pos=1, start=0, stop=75, label="JetVBFLeadingPtNotFromHiggs_mass1"),
        **create_HistConf("JetVBFLeadingPtNotFromHiggs", "btagPNetB", bins=100, pos=1, start=0, stop=1, label="JetVBFLeadingPtNotFromHiggs_btagPNetB1"),
        **create_HistConf("JetVBFLeadingPtNotFromHiggs", "btagPNetQvG", bins=100, pos=1, start=0, stop=1, label="JetVBFLeadingPtNotFromHiggs_btagPNetQvG1"),

        **create_HistConf("JetVBFLeadingMjjNotFromHiggs", "pt", bins=100, pos=0, start=0, stop=700, label="JetVBFLeadingMjjNotFromHiggs_pt0"),
        **create_HistConf("JetVBFLeadingMjjNotFromHiggs", "eta", bins=60, pos=0, start=-5, stop=5, label="JetVBFLeadingMjjNotFromHiggs_eta0"),
        **create_HistConf("JetVBFLeadingMjjNotFromHiggs", "phi", bins=60, pos=0, start=-5, stop=5, label="JetVBFLeadingMjjNotFromHiggs_phi0"),
        **create_HistConf("JetVBFLeadingMjjNotFromHiggs", "mass", bins=100, pos=0, start=0, stop=75, label="JetVBFLeadingMjjNotFromHiggs_mass0"),
        **create_HistConf("JetVBFLeadingMjjNotFromHiggs", "btagPNetB", bins=100, pos=0, start=0, stop=1, label="JetVBFLeadingMjjNotFromHiggs_btagPNetB0"),
        **create_HistConf("JetVBFLeadingMjjNotFromHiggs", "btagPNetQvG", bins=100, pos=0, start=0, stop=1, label="JetVBFLeadingMjjNotFromHiggs_btagPNetQvG0"),
        **create_HistConf("JetVBFLeadingMjjNotFromHiggs", "pt", bins=100, pos=1, start=0, stop=700, label="JetVBFLeadingMjjNotFromHiggs_pt1"),
        **create_HistConf("JetVBFLeadingMjjNotFromHiggs", "eta", bins=60, pos=1, start=-5, stop=5, label="JetVBFLeadingMjjNotFromHiggs_eta1"),
        **create_HistConf("JetVBFLeadingMjjNotFromHiggs", "phi", bins=60, pos=1, start=-5, stop=5, label="JetVBFLeadingMjjNotFromHiggs_phi1"),
        **create_HistConf("JetVBFLeadingMjjNotFromHiggs", "mass", bins=100, pos=1, start=0, stop=75, label="JetVBFLeadingMjjNotFromHiggs_mass1"),
        **create_HistConf("JetVBFLeadingMjjNotFromHiggs", "btagPNetB", bins=100, pos=1, start=0, stop=1, label="JetVBFLeadingMjjNotFromHiggs_btagPNetB1"),
        **create_HistConf("JetVBFLeadingMjjNotFromHiggs", "btagPNetQvG", bins=100, pos=1, start=0, stop=1, label="JetVBFLeadingMjjNotFromHiggs_btagPNetQvG1"),

        **create_HistConf("events", "JetVBFLeadingPtNotFromHiggs_deltaEta", bins=11, start=0, stop=10, label="JetVBFLeadingPtNotFromHiggs_deltaEta"),
        **create_HistConf("events", "JetVBFLeadingMjjNotFromHiggs_deltaEta", bins=11, start=0, stop=10, label="JetVBFLeadingMjjNotFromHiggs_deltaEta"),
        **create_HistConf("events", "JetVBFLeadingPtNotFromHiggs_jjMass", bins=100, start=0, stop=2000, label="JetVBFLeadingPtNotFromHiggs_jjMass"),
        **create_HistConf("events", "JetVBFLeadingMjjNotFromHiggs_jjMass", bins=100, start=0, stop=2000, label="JetVBFLeadingMjjNotFromHiggs_jjMass"),
    },
    columns={
        "common": {
            "inclusive": (
                [
                    ColOut(
                        "events",
                        [
                            "etaProduct",
                            "JetVBFLeadingPtNotFromHiggs_deltaEta",
                            "JetVBFLeadingMjjNotFromHiggs_deltaEta",
                            "JetVBFLeadingPtNotFromHiggs_jjMass",
                            "JetVBFLeadingMjjNotFromHiggs_jjMass",
                            "HH",
                            "HH_centrality",
                            "HH_deltaR",
                            "jj_deltaR",
                            "H1j1_deltaR",
                            "H1j2_deltaR",
                            "H2j1_deltaR",
                            "H2j2_deltaR",
                        ],
                    ),
                    ColOut(
                        "Jet",
                        jet_info,
                    ),
                    ColOut(
                        "JetVBFNotFromHiggs",
                        jet_info,
                    ),
                    ColOut(
                        "JetGoodFromHiggsOrdered",
                        jet_info,
                    ),
                    ColOut(
                        "JetVBF_matching",
                        jet_info,
                    ),
                    ColOut(
                        "JetVBFLeadingPtNotFromHiggs",
                        jet_info,
                    ),
                    ColOut(
                        "JetVBFLeadingMjjNotFromHiggs",
                        jet_info,
                    ),
                    ColOut(
                        "HH",
                        ["pt", "eta", "phi", "mass"],
                    ),
                ]
                + [
                    ColOut(
                        "quarkVBF_matched",
                        [
                            "index",
                            "pt",
                            "eta",
                            "phi",
                        ],
                    ),
                    ColOut(
                        "quarkVBF",
                        [
                            "index",
                            "pt",
                            "eta",
                            "phi",
                        ],
                    ),
                    ColOut(
                        "quarkVBF_generalSelection_matched",
                        [
                            "index",
                            "pt",
                            "eta",
                            "phi",
                        ],
                    ),
                    ColOut(
                        "JetVBF_matched",
                        jet_info,
                    ),
                    ColOut(
                        "JetVBF_generalSelection_matched",
                        jet_info,
                    ),
                    ColOut(
                        "events",
                        [
                            "deltaEta_matched",
                            "jj_mass_matched",
                            "nJetVBF_matched",
                        ],
                    ),
                ]
                if VBF_PARTON_MATCHING
                else [
                    ColOut(
                        "events",
                        [
                            "HH",
                            "JetVBFLeadingPtNotFromHiggs_deltaEta",
                            "JetVBFLeadingMjjNotFromHiggs_deltaEta",
                            "JetVBFLeadingPtNotFromHiggs_jjMass",
                            "JetVBFLeadingMjjNotFromHiggs_jjMass",
                            "HH_deltaR",
                            "H1j1_deltaR",
                            "H1j2_deltaR",
                            "H2j1_deltaR",
                            "H2j2_deltaR",
                            "HH_centrality",
                        ],
                    ),
                    ColOut(
                        "HiggsLeading",
                        ["pt", "eta", "phi", "mass"]
                    ),
                    ColOut(
                        "HiggsSubLeading",
                        ["pt", "eta", "phi", "mass"]
                    ),
                    ColOut(
                        "Jet",
                        jet_info,
                    ),
                    ColOut(
                        "JetGoodFromHiggsOrdered",
                        jet_info,
                    ),
                    ColOut(
                        "JetVBFLeadingPtNotFromHiggs",
                        jet_info,
                    ),
                    ColOut(
                        "JetVBFLeadingMjjNotFromHiggs",
                        jet_info,
                    ),
                    ColOut(
                        "HH",
                        ["pt", "eta", "phi", "mass"],
                    ),
                ]
            ),
        },
        "bysample": {},
    },
)