import os
import sys

from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.cut_functions import (
    get_HLTsel,
)
from pocket_coffea.parameters.histograms import *
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.lib.columns_manager import ColOut
from pocket_coffea.parameters import defaults
from pocket_coffea.lib.weights.common.common import common_weights

from workflow import VBFHH4bProcessor
from custom_cuts import vbf_hh4b_presel, vbf_hh4b_presel_tight

from configs.HH4b_common.custom_cuts_common import hh4b_2b_region, hh4b_4b_region
from configs.HH4b_common.custom_weights import bkg_morphing_dnn_weight
from configs.HH4b_common.configurator_options import (
    get_variables_dict,
    get_columns_list,
)

localdir = os.path.dirname(os.path.abspath(__file__))

# Loading default parameters

default_parameters = defaults.get_default_parameters()
defaults.register_configuration_dir("config_dir", localdir + "/params")

# adding object preselection
year = "2022_postEE"
parameters = defaults.merge_parameters_from_files(
    default_parameters,
    f"{localdir}/params/object_preselectionSemiTight.yaml",
    f"{localdir}/params/triggers.yaml",
    f"{localdir}/params/jets_calibration.yaml",
    # f"{localdir}/params/plotting_style.yaml",
    update=True,
)

SPANET_MODEL = (
    "params/out_hh4b_5jets_ATLAS_ptreg_c0_lr1e4_wp0_noklininp_oc_300e_kl3p5.onnx"
)
VBF_GGF_DNN_MODEL = "/t3home/rcereghetti/ML_pytorch/out/20241212_223142_SemitTightPtLearningRateConstant/models/model_28.onnx"
BKG_MORPHING_DNN_MODEL = "/pnfs/psi.ch/cms/trivcat/store/user/mmalucch/keras_models_morphing/average_model_from_keras.onnx"
VBF_GGF_DNN_MODEL = ""
# BKG_MORPHING_DNN_MODEL=""

HIGGS_PARTON_MATCHING = False
VBF_PARTON_MATCHING = False
TIGHT_CUTS = False
CLASSIFICATION = False
SAVE_CHUNK = False

workflow_options = {
    "parton_jet_min_dR": 0.4,
    "max_num_jets": 5,
    "which_bquark": "last",
    "classification": CLASSIFICATION,  # HERE
    "SPANET_MODEL": SPANET_MODEL,
    "BKG_MORPHING_DNN_MODEL": BKG_MORPHING_DNN_MODEL,
    "VBF_GGF_DNN_MODEL": VBF_GGF_DNN_MODEL,
    "tight_cuts": TIGHT_CUTS,
    "fifth_jet": "pt",
    "donotscale_sumgenweights": True,
    "vbf_parton_matching": VBF_PARTON_MATCHING,
}
if SAVE_CHUNK:
    # workflow_options["dump_columns_as_arrays_per_chunk"] = "root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/tharte/HH4b/training_samples/GluGlutoHHto4B_spanet_loose_03_17"
    pass

jet_info = ["index", "pt", "btagPNetQvG", "eta", "btagPNetB", "phi", "mass"]

variables_dict = get_variables_dict(CLASSIFICATION, False, False)

columns_dict = {
    "HiggsLeading": ["pt", "mass", "dR"],
    "HiggsSubLeading": ["pt", "mass", "dR"],
    "HH": ["mass"],
    "events": ["dR_min", "dR_max", "bkg_morphing_dnn_weight"],
}

column_list = get_columns_list(columns_dict, SAVE_CHUNK)


cfg = Configurator(
    parameters=parameters,
    datasets={
        "jsons": [
            f"{localdir}/datasets/QCD.json",
            f"{localdir}/datasets/signal_VBF_HH4b.json",
            f"{localdir}/datasets/signal_ggF_HH4b_local.json",
            f"{localdir}/datasets/DATA_JetMET_skimmed.json",
        ],
        "filter": {
            "samples": (
                [
                    # "VBF_HHto4B",
                    "DATA_JetMET_JMENano_skimmed",
                    # "GluGlutoHHto4B",
                ]
            ),
            "samples_exclude": [],
            "year": [year],
        },
        "subsamples": {},
    },
    workflow=VBFHH4bProcessor,
    workflow_options=workflow_options,
    skim=[
        get_HLTsel(primaryDatasets=["JetMET"]),
    ],
    preselections=[vbf_hh4b_presel if TIGHT_CUTS is False else vbf_hh4b_presel_tight],
    categories={
        **{"4b_region": [hh4b_4b_region]},
        **{"2b_region": [hh4b_2b_region]},
        # **{f"4b_semiTight_LeadingPt_region": [hh4b_4b_region, semiTight_leadingPt]},
        # **{f"4b_semiTight_LeadingMjj_region": [hh4b_4b_region, semiTight_leadingMjj]},
        # **{f"4b_semiTight_LeadingMjj_region": [hh4b_4b_region, semiTight_leadingMjj]}
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
    },
    weights_classes=common_weights + [bkg_morphing_dnn_weight],
    weights={
        "common": {
            "inclusive": [
                "genWeight",
                "lumi",
                "XS",
            ],
            "bycategory": {},
        },
        "bysample": {
            "DATA_JetMET_JMENano_skimmed": {
                "inclusive": [],
                "bycategory": {
                    "2b_region": ["bkg_morphing_dnn_weight"],
                },
            },
        },
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
    variables=variables_dict,
    columns={
        "common": {
            "inclusive": (
                [
                    # ColOut("events", ["bkg_morphing_dnn_weight"])
                    #             "etaProduct",
                    #             "JetVBFLeadingPtNotFromHiggs_deltaEta",
                    #             "JetVBFLeadingMjjNotFromHiggs_deltaEta",
                    #             "JetVBFLeadingPtNotFromHiggs_jjMass",
                    #             "JetVBFLeadingMjjNotFromHiggs_jjMass",
                    #             "HH",
                    #             "HH_centrality",
                    #             "HH_deltaR",
                    #             "jj_deltaR",
                    #             "H1j1_deltaR",
                    #             "H1j2_deltaR",
                    #             "H2j1_deltaR",
                    #             "H2j2_deltaR",
                    #         ],
                    #     ),
                    #     ColOut(
                    #         "Jet",
                    #         jet_info,
                    #     ),
                    #     ColOut(
                    #         "JetVBFNotFromHiggs",
                    #         jet_info,
                    #     ),
                    #     ColOut(
                    #         "JetGoodFromHiggsOrdered",
                    #         jet_info,
                    #     ),
                    #     ColOut(
                    #         "JetVBF_matching",
                    #         jet_info,
                    #     ),
                    #     ColOut(
                    #         "JetVBFLeadingPtNotFromHiggs",
                    #         jet_info,
                    #     ),
                    #     ColOut(
                    #         "JetVBFLeadingMjjNotFromHiggs",
                    #         jet_info,
                    #     ),
                    #     ColOut(
                    #         "HH",
                    #         ["pt", "eta", "phi", "mass"],
                    #     ),
                    # ]
                    # + [
                    #     ColOut(
                    #         "quarkVBF_matched",
                    #         [
                    #             "index",
                    #             "pt",
                    #             "eta",
                    #             "phi",
                    #         ],
                    #     ),
                    #     ColOut(
                    #         "quarkVBF",
                    #         [
                    #             "index",
                    #             "pt",
                    #             "eta",
                    #             "phi",
                    #         ],
                    #     ),
                    #     ColOut(
                    #         "quarkVBF_generalSelection_matched",
                    #         [
                    #             "index",
                    #             "pt",
                    #             "eta",
                    #             "phi",
                    #         ],
                    #     ),
                    #     ColOut(
                    #         "JetVBF_matched",
                    #         jet_info,
                    #     ),
                    #     ColOut(
                    #         "JetVBF_generalSelection_matched",
                    #         jet_info,
                    #     ),
                    #     ColOut(
                    #         "events",
                    #         [
                    #             "deltaEta_matched",
                    #             "jj_mass_matched",
                    #             "nJetVBF_matched",
                    #         ],
                    #     ),
                    # ]
                    # if VBF_PARTON_MATCHING
                    # else [
                    #     ColOut(
                    #         "events",
                    #         [
                    #             "HH",
                    #             "JetVBFLeadingPtNotFromHiggs_deltaEta",
                    #             "JetVBFLeadingMjjNotFromHiggs_deltaEta",
                    #             "JetVBFLeadingPtNotFromHiggs_jjMass",
                    #             "JetVBFLeadingMjjNotFromHiggs_jjMass",
                    #             "HH_deltaR",
                    #             "H1j1_deltaR",
                    #             "H1j2_deltaR",
                    #             "H2j1_deltaR",
                    #             "H2j2_deltaR",
                    #             "HH_centrality",
                    #         ],
                    #     ),
                    #     ColOut(
                    #         "HiggsLeading",
                    #         ["pt", "eta", "phi", "mass"]
                    #     ),
                    #     ColOut(
                    #         "HiggsSubLeading",
                    #         ["pt", "eta", "phi", "mass"]
                    #     ),
                    #     ColOut(
                    #         "Jet",
                    #         jet_info,
                    #     ),
                    #     ColOut(
                    #         "JetGoodFromHiggsOrdered",
                    #         jet_info,
                    #     ),
                    #     ColOut(
                    #         "JetVBFLeadingPtNotFromHiggs",
                    #         jet_info,
                    #     ),
                    #     ColOut(
                    #         "JetVBFLeadingMjjNotFromHiggs",
                    #         jet_info,
                    #     ),
                    #     ColOut(
                    #         "HH",
                    #         ["pt", "eta", "phi", "mass"],
                    #     ),
                ]
            ),
        },
        "bysample": {
            "DATA_JetMET_JMENano_skimmed": {
                "inclusive": column_list,
            },
        },
    },
)
