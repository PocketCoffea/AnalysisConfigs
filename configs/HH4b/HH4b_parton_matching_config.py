# for spanet evaluation: pocket-coffea run --cfg HH4b_parton_matching_config.py -e dask@T3_CH_PSI --custom-run-options params/t3_run_options.yaml -o /work/mmalucch/out_test --executor-custom-setup onnx_executor.py

from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.cut_functions import (
    get_nObj_eq,
    get_nObj_min,
    get_nObj_less,
    get_HLTsel,
    get_nBtagMin,
    get_nElectron,
    get_nMuon,
)
from pocket_coffea.parameters.histograms import *
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.lib.columns_manager import ColOut

import workflow
from workflow import HH4bbQuarkMatchingProcessor

import custom_cut_functions
import custom_cuts
from custom_cut_functions import *
from custom_cuts import *


import os

localdir = os.path.dirname(os.path.abspath(__file__))

# Loading default parameters
from pocket_coffea.parameters import defaults

CLASSIFICATION = False
TIGHT_CUTS=False

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

# spanet_model="params/spanet_5jets_ptreg_ATLAS.onnx"
spanet_model = (
    "params/out_hh4b_5jets_ATLAS_ptreg_c0_lr1e4_wp0_noklininp_oc_300e_kl3p5.onnx"
)

cfg = Configurator(
    parameters=parameters,
    datasets={
        "jsons": [
            #f"{localdir}/datasets/DATA_JetMET.json",
            #f"{localdir}/datasets/QCD.json",
            #f"{localdir}/datasets/SPANet_classification.json",
            f"{localdir}/datasets/signal_ggF_HH4b.json",
        ],
        "filter": {
            "samples": (
                [
                    #"GluGlutoHHto4B",
                    # "QCD-4Jets",
                    # "DATA_JetMET_JMENano",

                    # "SPANet_classification",
                    # "SPANet_classification_data",
                    # "GluGlutoHHto4B_poisson",
                    # "GluGlutoHHto4B_private",
                    "GluGlutoHHto4B_spanet",
                ]
                if CLASSIFICATION
                else ["GluGlutoHHto4B_spanet"]
            ),
            "samples_exclude": [],
            "year": [year],
        },
        "subsamples": {},
    },
    workflow=HH4bbQuarkMatchingProcessor,
    workflow_options={
        "parton_jet_min_dR": 0.4,
        "max_num_jets": 5,
        "which_bquark": "last",
        "classification": CLASSIFICATION,  # HERE
        "spanet_model": spanet_model,
        "fifth_jet" : "pt",
    },
    skim=[
        get_HLTsel(primaryDatasets=["JetMET"]),
    ],
    preselections=[
        # lepton_veto_presel,
        # four_jet_presel,
        # jet_pt_presel,
        # jet_btag_presel,
        hh4b_presel
    ],
    categories={
        # "lepton_veto": [lepton_veto_presel],
        # "four_jet": [four_jet_presel],
        # "jet_pt": [jet_pt_presel],
        # "jet_btag_lead": [jet_btag_lead_presel],
        # "jet_pt_copy": [jet_pt_presel],
        # "jet_btag_medium": [jet_btag_medium_presel],
        # "jet_pt_copy2": [jet_pt_presel],
        # "jet_btag_loose": [jet_btag_loose_presel],
        # "full_parton_matching": [
        #     jet_btag_medium,
        #     get_nObj_eq(4, coll="bQuarkHiggsMatched"),
        # ],
        "4b_region": [hh4b_4b_region],  # HERE
        "2b_region": [hh4b_2b_region],
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
        # **count_hist(coll="JetGoodHiggs", bins=10, start=0, stop=10),
        # **count_hist(coll="ElectronGood", bins=3, start=0, stop=3),
        # **count_hist(coll="MuonGood", bins=3, start=0, stop=3),
        # **count_hist(coll="JetGoodHiggsMatched", bins=10, start=0, stop=10),
        # **count_hist(coll="bQuarkHiggsMatched", bins=10, start=0, stop=10),
        # **count_hist(coll="JetGoodMatched", bins=10, start=0, stop=10),
        # **count_hist(coll="bQuarkMatched", bins=10, start=0, stop=10),
        # **jet_hists(coll="JetGood", pos=0),
        # **jet_hists(coll="JetGood", pos=1),
        # **jet_hists(coll="JetGood", pos=2),
        # **jet_hists(coll="JetGood", pos=3),
        # **jet_hists(coll="JetGoodHiggsPtOrder", pos=0),
        # **jet_hists(coll="JetGoodHiggsPtOrder", pos=1),
        # **jet_hists(coll="JetGoodHiggsPtOrder", pos=2),
        # **jet_hists(coll="JetGoodHiggsPtOrder", pos=3),
        # **jet_hists(coll="JetGoodHiggs", pos=0),
        # **jet_hists(coll="JetGoodHiggs", pos=1),
        # **jet_hists(coll="JetGoodHiggs", pos=2),
        # **jet_hists(coll="JetGoodHiggs", pos=3),
        # **parton_hists(coll="bQuarkHiggsMatched", pos=0),
        # **parton_hists(coll="bQuarkHiggsMatched", pos=1),
        # **parton_hists(coll="bQuarkHiggsMatched", pos=2),
        # **parton_hists(coll="bQuarkHiggsMatched", pos=3),
        # **parton_hists(coll="bQuarkHiggsMatched"),
        # **parton_hists(coll="JetGoodHiggsMatched", pos=0),
        # **parton_hists(coll="JetGoodHiggsMatched", pos=1),
        # **parton_hists(coll="JetGoodHiggsMatched", pos=2),
        # **parton_hists(coll="JetGoodHiggsMatched", pos=3),
        # **parton_hists(coll="JetGoodHiggsMatched"),
        # **parton_hists(coll="bQuarkMatched", pos=0),
        # **parton_hists(coll="bQuarkMatched", pos=1),
        # **parton_hists(coll="bQuarkMatched", pos=2),
        # **parton_hists(coll="bQuarkMatched", pos=3),
        # **parton_hists(coll="bQuarkMatched"),
        # **parton_hists(coll="JetGoodMatched", pos=0),
        # **parton_hists(coll="JetGoodMatched", pos=1),
        # **parton_hists(coll="JetGoodMatched", pos=2),
        # **parton_hists(coll="JetGoodMatched", pos=3),
        # **parton_hists(coll="JetGoodMatched"),
        # **{
        #     f"RecoHiggs1Mass": HistConf(
        #         [
        #             Axis(
        #                 coll=f"events",
        #                 field="RecoHiggs1Mass",
        #                 bins=30,
        #                 start=60,
        #                 stop=200,
        #                 label=f"RecoHiggs1Mass",
        #             )
        #         ]
        #     )
        # },
        # **{
        #     f"RecoHiggs2Mass": HistConf(
        #         [
        #             Axis(
        #                 coll=f"events",
        #                 field="RecoHiggs2Mass",
        #                 bins=30,
        #                 start=60,
        #                 stop=200,
        #                 label=f"RecoHiggs2Mass",
        #             )
        #         ]
        #     )
        # },
        # **{

    },
    columns={
        "common": {
            "inclusive": (
                [
                    # ColOut(
                    #     "bQuarkHiggsMatched",
                    #     [
                    #         "provenance",
                    #         "dRMatchedJet",
                    #         "pt",
                    #         "eta",
                    #         "phi",
                    #         "mass",
                    #     ],
                    # ),
                    # ColOut(
                    #     "bQuarkMatched",
                    #     [
                    #         "provenance",
                    #         "dRMatchedJet",
                    #         "pt",
                    #         "eta",
                    #         "phi",
                    #         "mass",
                    #     ],
                    # ),
                    # ColOut(
                    #     "bQuark",
                    #     [
                    #         "provenance",
                    #         "pt",
                    #         "eta",
                    #         "phi",
                    #         "mass",
                    #     ],
                    # ),
                    ColOut(
                        "JetGoodHiggsMatched",
                        [
                            "provenance",
                            # "pdgId",
                            # "dRMatchedJet",
                            "pt",
                            "eta",
                            "phi",
                            # "cosPhi",
                            # "sinPhi",
                            "mass",
                            "btagPNetB",
                            # "hadronFlavour",
                        ],
                    ),
                    ColOut(
                        "JetGoodMatched",
                        [
                            "provenance",
                            # "pdgId",
                            # "dRMatchedJet",
                            "pt",
                            "eta",
                            "phi",
                            # "cosPhi",
                            # "sinPhi",
                            "mass",
                            "btagPNetB",
                            # "hadronFlavour",
                        ],
                    ),
                    ColOut(
                        "JetGoodHiggs",
                        [
                            "provenance",
                            "pt",
                            "eta",
                            "phi",
                            # "cosPhi",
                            # "sinPhi",
                            "mass",
                            "btagPNetB",
                            # "hadronFlavour",
                        ],
                    ),
                    ColOut(
                        "JetGood",
                        [
                            "provenance",
                            "pt",
                            "eta",
                            "phi",
                            # "cosPhi",
                            # "sinPhi",
                            "mass",
                            "btagPNetB",
                            # "hadronFlavour",
                        ],
                    ),
                ]
                + (
                    [
                        ColOut(
                            "HiggsLeading",
                            [
                                "pt",
                                "eta",
                                "phi",
                                "mass",
                                "dR",
                                "cos_theta",
                            ],
                        ),
                        ColOut(
                            "HiggsSubLeading",
                            [
                                "pt",
                                "eta",
                                "phi",
                                "mass",
                                "dR",
                                "cos_theta",
                            ],
                        ),
                        ColOut(
                            "HH",
                            [
                                "pt",
                                "eta",
                                # "phi",
                                "mass",
                                "cos_theta_star",
                                "dR",
                                "dPhi",
                                "dEta",
                            ],
                        ),
                        ColOut(
                            "events",
                            [
                                "best_pairing_probability",
                                "second_best_pairing_probability",
                                "Delta_pairing_probabilities",
                                "HT",
                                "dR_min",
                                "dR_max",
                            ],
                        ),
                    ]
                    if CLASSIFICATION
                    else []
                )
            ),
        },
        "bysample": {},
    },
)

run_options = {
    "executor": "dask/slurm",
    "env": "conda",
    "workers": 1,
    "scaleout": 200,
    "worker_image": "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/pocketcoffea:lxplus-cc7-latest",
    "queue": "standard",
    "walltime": "00:20:00",
    "mem_per_worker": "6GB",  # GB
    "disk_per_worker": "1GB",  # GB
    "exclusive": False,
    "chunk": 100000,
    "retries": 50,
    "treereduction": 5,
    "adapt": False,
}


if "dask" in run_options["executor"]:
    import cloudpickle

    cloudpickle.register_pickle_by_value(workflow)
    cloudpickle.register_pickle_by_value(custom_cut_functions)
    cloudpickle.register_pickle_by_value(custom_cuts)
