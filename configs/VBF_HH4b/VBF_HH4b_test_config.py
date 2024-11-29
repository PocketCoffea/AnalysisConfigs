from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.cut_functions import (
    get_HLTsel,
)
from pocket_coffea.parameters.histograms import *
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.lib.columns_manager import ColOut

from workflow import VBFHH4bbQuarkMatchingProcessor
from custom_cut_functions import *
from custom_cuts import *

import os

localdir = os.path.dirname(os.path.abspath(__file__))

# Loading default parameters
from pocket_coffea.parameters import defaults

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

# spanet_model = (
#     "params/out_hh4b_5jets_ATLAS_ptreg_c0_lr1e4_wp0_noklininp_oc_300e_kl3p5.onnx"
# )
spanet_model = None

# TODO: spostare queste funzioni?

# Combine jet_hists from position start to position end
def jet_hists_dict(coll="JetGood", start=1, end=5):
    """Combina i dizionari creati da jet_hists per ogni posizione da start a stop (inclusi)."""
    combined_dict = {}
    for pos in range(start, end + 1):
        combined_dict.update(jet_hists(coll=coll, pos=pos))  # Unisce ogni dizionario al precedente
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
                    # "GluGlutoHHto4B",
                    #TODO qcd
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
        "spanet_model": spanet_model,
    },
    skim=[
        get_HLTsel(primaryDatasets=["JetMET"]),
    ],
    preselections=[
        hh4b_presel
    ],
    categories={
        **{"4b_region": [hh4b_4b_region]},
        # **{"4b_VBFtight_region": [hh4b_4b_region, VBFtight_region]},
        # **{"4b_VBFtight_region": [hh4b_4b_region, vbf_wrapper()]},
        # HERE **{f"4b_VBFtight_{list(ab[0].keys())[i]}_region": [hh4b_4b_region, vbf_wrapper(ab[i])] for i in range(0, 1)},
        # **{"4b_VBF_generalSelection_region": [hh4b_4b_region, VBF_generalSelection_region]},
        # **{"4b_VBF_region": [hh4b_4b_region, VBF_region]},
        # **{f"4b_VBF_0{i}qvg_region": [hh4b_4b_region, VBF_region, qvg_regions[f"qvg_0{i}_region"]] for i in range(5, 10)},
        # **{f"4b_VBF_0{i}qvg_generalSelection_region": [hh4b_4b_region, VBF_generalSelection_region, qvg_regions[f"qvg_0{i}_region"]] for i in range(5, 10)},
        # "2b_region": [hh4b_2b_region],
        #TODO
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

        **create_HistConf("JetGoodVBF_matched", "eta", bins=60, start=-5, stop=5, label="JetVBF_matched_eta"),
        **create_HistConf("events", "etaProduct", bins=5, start=-2.5, stop=2.5, label="JetVBF_matched_eta_product"),
        **create_HistConf("JetGoodVBF_matched", "pt", bins=100, start=0, stop=1000, label="JetVBF_matched_pt"),
        **create_HistConf("JetGoodVBF_matched", "btagPNetQvG", pos=0, bins=60, start=0, stop=1, label="JetVBF_matchedQvG_0"),
        **create_HistConf("JetGoodVBF_matched", "btagPNetQvG", pos=1, bins=60, start=0, stop=1, label="JetVBF_matchedQvG_1"),
        **create_HistConf("quarkVBF_matched", "eta", bins=60, start=-5, stop=5, label="quarkVBF_matched_Eta"),
        **create_HistConf("quarkVBF_matched", "pt", bins=100, start=0, stop=1000, label="quarkVBF_matched_pt"),
        **create_HistConf("JetGoodVBF_matched", "btagPNetB", bins=100, start=0, stop=1, label="JetGoodVBF_matched_btag"),
        **create_HistConf("events", "deltaEta_matched", bins=100, start=0, stop=10, label="deltaEta"),
        **create_HistConf("events", "jj_mass_matched", bins=100, start=0, stop=5000, label="jj_mass"),
        **create_HistConf("events", "HH_mass", bins=100, start=0, stop=2500, label="HH_mass"),
    },
    columns={
        "common": {
            "inclusive": (
                [
                    ColOut(
                        "events",
                        [
                            "etaProduct",
                            "deltaEta_matched",
                            "jj_mass_matched",
                            "HH_mass",
                        ],
                    ),
                    ColOut(
                        "Jet",
                        [
                            "index"
                        ],
                    ),
                    ColOut(
                        "JetGoodVBF_matched",
                        [
                            "index",
                            "pt",
                            "btagPNetQvG",
                            "eta",
                        ],
                    ),
                    ColOut(
                        "quarkVBF_matched",
                        [
                            "index",
                            "pt",
                            "eta",
                        ],
                    ),
                    ColOut(
                        "JetGoodMatched",
                        [
                            "index",
                        ],
                    ),
                ]

            ),
        },
        "bysample": {},
    },
)
