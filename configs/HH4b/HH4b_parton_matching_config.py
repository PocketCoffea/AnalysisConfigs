# for spanet evaluation: pocket-coffea run --cfg HH4b_parton_matching_config.py -e dask@T3_CH_PSI --custom-run-options params/t3_run_options.yaml -o /work/mmalucch/out_test --executor-custom-setup onnx_executor.py

from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.cut_functions import (
    get_HLTsel,
)
from pocket_coffea.parameters.histograms import *
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.lib.columns_manager import ColOut
from pocket_coffea.parameters import defaults

from workflow import HH4bbQuarkMatchingProcessor
from custom_cut_functions import *
from custom_cuts import *

import os

localdir = os.path.dirname(os.path.abspath(__file__))

# Loading default parameters

CLASSIFICATION = False
TIGHT_CUTS = False
RANDOM_PT = True
SAVE_CHUNK = True

print("CLASSIFICATION ", CLASSIFICATION)
print("TIGHT_CUTS ", TIGHT_CUTS)
print("RANDOM_PT ", RANDOM_PT)

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
    "/work/tharte/datasets/mass_sculpting_data/hh4b_5jets_e300_s101_no_btag.onnx"
#    "/work/tharte/datasets/mass_sculpting_data/hh4b_5jets_e300_s100_ptvary_loose_btag.onnx"
#    "/work/tharte/datasets/mass_sculpting_data/hh4b_5jets_e300_s100_ptvary_tight_btag.onnx"
#    "/work/tharte/datasets/mass_sculpting_data/hh4b_5jets_e300_s100_ptvary_wide_loose_btag.onnx"
#    "/work/tharte/datasets/mass_sculpting_data/hh4b_5jets_e300_s100_ptvary_wide_onlylog_loose_btag.onnx"
#    "/work/tharte/datasets/mass_sculpting_data/hh4b_5jets_e300_s100_ptvary_01_10_loose_btag.onnx"
#    "/work/tharte/datasets/mass_sculpting_data/hh4b_5jets_e300_s100_ptnone_loose_btag.onnx"
#    "/work/tharte/datasets/mass_sculpting_data/hh4b_5jets_e300_s100_ptvary_wide_tight_btag.onnx"
#    "/work/tharte/datasets/mass_sculpting_data/hh4b_5jets_e300_s160_btag.onnx"
#    "/work/tharte/datasets/mass_sculpting_data/out_hh4b_5jets_ATLAS_ptreg_c0_lr1e4_wp0_noklininp_oc_300e_kl3p5.onnx"
)

workflow_options ={
        "parton_jet_min_dR": 0.4,
        "max_num_jets": 5,
        "which_bquark": "last",
        "classification": CLASSIFICATION,  # HERE
        "spanet_model": SPANET_MODEL,
        "tight_cuts": TIGHT_CUTS,
        "fifth_jet": "pt",
        "random_pt": RANDOM_PT,
        "rand_type": 0.3
    }
if SAVE_CHUNK:
    workflow_options["dump_columns_as_arrays_per_chunk"] = "root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/tharte/HH4b/training_samples/GluGlutoHHto4B_spanet_loose_03_17"
    # workflow_options["dump_columns_as_arrays_per_chunk"] = "root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/tharte/HH4b/ntuples/DATA_JetMET_JMENano_btag_ordering"
    # workflow_options["dump_columns_as_arrays_per_chunk"] = "root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/tharte/HH4b/ntuples/DATA_JetMET_JMENano_no_btag"


variables_dict = {
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
        **jet_hists(coll="JetGoodHiggs", pos=0),
        **jet_hists(coll="JetGoodHiggs", pos=1),
        **jet_hists(coll="JetGoodHiggs", pos=2),
        **jet_hists(coll="JetGoodHiggs", pos=3),
        # **parton_hists(coll="bQuarkHiggsMatched", pos=0),
        # **parton_hists(coll="bQuarkHiggsMatched", pos=1),
        # **parton_hists(coll="bQuarkHiggsMatched", pos=2),
        # **parton_hists(coll="bQuarkHiggsMatched", pos=3),
        # **parton_hists(coll="bQuarkHiggsMatched"),
        # **parton_hists(coll="JetGoodHiggsMatched", pos=0),
        # **parton_hists(coll="JetGoodHiggsMatched", pos=1),
        # **parton_hists(coll="JetGoodHiggsMatched", pos=2),
        # **parton_hists(coll="JetGoodHiggsMatched", pos=3),
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
        }
if CLASSIFICATION:
    variables_dict.update({
        "RecoHiggs1Mass": HistConf(
            [
                Axis(
                    coll=f"HiggsLeading",
                    field="mass",
                    bins=240,
                    start=0,
                    stop=240,
                    label=r"$M_{H_1}$ SPANet",
                )
            ],

        ),
        "RecoHiggs1Mass_Dhh": HistConf(
            [
                Axis(
                    coll=f"HiggsLeadingRun2",
                    field="mass",
                    bins=240,
                    start=0,
                    stop=240,
                    label=r"$M_{H_1}$ $D_{HH}$",
                )
            ],
        ),
        "RecoHiggs2Mass": HistConf(
            [
                Axis(
                    coll=f"HiggsSubLeading",
                    field="mass",
                    bins=240,
                    start=0,
                    stop=240,
                    label=r"$M_{H_2}$ SPANet",
                )
            ],
        ),
        "RecoHiggs2Mass_Dhh": HistConf(
            [
                Axis(
                    coll=f"HiggsSubLeadingRun2",
                    field="mass",
                    bins=240,
                    start=0,
                    stop=240,
                    label=r"$M_{H_2}$ $D_{HH}$",
                )
            ],
        )
    })
if RANDOM_PT:
    variables_dict.update({
        "Random_pt_Factor": HistConf(
            [
                Axis(
                    coll=f"events",
                    field="random_pt_weights",
                    bins=50,
                    start=0,
                    stop=2,
                    label=r"$pT$",
                )
            ],
        ),
    })


cfg = Configurator(
    # save_skimmed_files="root://t3dcachedb03.psi.ch:1094//pnfs/psi.ch/cms/trivcat/store/user/tharte/HH4b/ntuples/DATA_JetMET_JMENano_skimmed",
    parameters=parameters,
    datasets={
        "jsons": [
            # f"{localdir}/datasets/DATA_JetMET.json",
            # f"{localdir}/datasets/QCD.json",
            # f"{localdir}/datasets/SPANet_classification.json",
            f"{localdir}/datasets/signal_ggF_HH4b.json",
            f"{localdir}/datasets/DATA_JetMET.json",
            f"{localdir}/datasets/DATA_JetMET_skimmed.json",
            f"{localdir}/datasets/QCD.json",
            f"{localdir}/datasets/SPANet_classification.json",
            f"{localdir}/datasets/signal_ggF_HH4b_local.json",
            f"{localdir}/datasets/signal_VBF_HH4b_local.json",
        ],
        "filter": {
            "samples": (
                [
                    # "GluGlutoHHto4B",
                    # "QCD-4Jets",
                    #"DATA_JetMET_JMENano",
                    "DATA_JetMET_JMENano_skimmed",
                    # "SPANet_classification",
                    # "SPANet_classification_data",
                    # "GluGlutoHHto4B_poisson",
                    # "GluGlutoHHto4B_private",
                    #"GluGlutoHHto4B_spanet",
                ]
                if CLASSIFICATION
                # else ["DATA_JetMET_JMENano"]
                else ["GluGlutoHHto4B_spanet"]
                # else ["GluGlutoHHto4B"]
            ),
            "samples_exclude": [],
            "year": [year],
        },
        "subsamples": {},
    },
    workflow=HH4bbQuarkMatchingProcessor,
    workflow_options=workflow_options,
    skim=[
        get_HLTsel(primaryDatasets=["JetMET"]),
    ],
    preselections=[
        # lepton_veto_presel,
        # four_jet_presel,
        # jet_pt_presel,
        # jet_btag_presel,
        hh4b_presel if TIGHT_CUTS == False else hh4b_presel_tight
    ],
    categories={
        # "baseline":[passthrough],
        # "lepton_veto": [lepton_veto_presel],
        # "four_jet": [four_jet_presel],
        # "jet_pt": [jet_pt_presel],
        # "jet_btag_lead": [jet_btag_lead_presel],
        # "jet_pt_copy": [jet_pt_presel],
        # "jet_btag_medium": [jet_btag_medium_presel],
        # "jet_pt_copy2": [jet_pt_presel],
        # "jet_btag_loose": [jet_btag_loose_presel],
        # "full_parton_matching": [
        #    jet_btag_medium,
        #     get_nObj_eq(4, coll="bQuarkHiggsMatched"),
        # ],
        "4b_region": [hh4b_4b_region],  # HERE
        #"4b_delta_Dhh_above_30": [hh4b_4b_region, dhh_above_30],
        "2b_region": [hh4b_2b_region],
        #"2b_delta_Dhh_above_30": [hh4b_2b_region, dhh_above_30],
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
    variables = variables_dict,
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
                    #     flatten=not SAVE_CHUNK,
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
                    #     flatten=not SAVE_CHUNK,
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
                    #     flatten=not SAVE_CHUNK,
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
                        flatten=not SAVE_CHUNK,
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
                        flatten=not SAVE_CHUNK,
                    ),
                    ColOut(
                        "JetGoodHiggs",
                        [
                            "provenance",
                            "pt",
                            "pt_orig",
                            "eta",
                            "phi",
                            # "cosPhi",
                            # "sinPhi",
                            "mass",
                            "mass_orig",
                            "btagPNetB",
                            # "hadronFlavour",
                        ],
                        flatten=not SAVE_CHUNK,
                    ),
                    ColOut(
                        "JetGood",
                        [
                            "provenance",
                            "pt",
                            "pt_orig",
                            "eta",
                            "phi",
                            # "cosPhi",
                            # "sinPhi",
                            "mass",
                            "mass_orig",
                            "btagPNetB",
                            # "hadronFlavour",
                        ],
                        flatten=not SAVE_CHUNK,
                    ),
                    ]
                    + ([
                        #ColOut(
                        #    "HiggsLeading",
                        #    [
                        #        "pt",
                        #        "eta",
                        #        "phi",
                        #        "mass",
                        #        # "dR",
                        #        # "cos_theta",
                        #    ],
                        #flatten=not SAVE_CHUNK,
                        #),
                        #ColOut(
                        #    "HiggsSubLeading",
                        #    [
                        #        "pt",
                        #        "eta",
                        #        "phi",
                        #        "mass",
                        #        # "dR",
                        #        # "cos_theta",
                        #    ],
                        #flatten=not SAVE_CHUNK,
                        #),
                        # ColOut(
                        #     "HiggsLeadingRun2",
                        #     [
                        #         "pt",
                        #         "eta",
                        #         "phi",
                        #         "mass",
                        #         # "dR",
                        #         # "cos_theta",
                        #     ],
                        #flatten=not SAVE_CHUNK,
                        # ),
                        # ColOut(
                        #     "HiggsSubLeadingRun2",
                        #     [
                        #         "pt",
                        #         "eta",
                        #         "phi",
                        #         "mass",
                        #         # "dR",
                        #         # "cos_theta",
                        #     ],
                        #flatten=not SAVE_CHUNK,
                        # ),
                       # ColOut(
                       #     "JetGoodFromHiggsOrderedRun2",
                       #     [
                       #         "pt",
                       #         "eta",
                       #         "phi",
                       #         "mass",
                       #         "btagPNetB",
                       #     ],
                       #flatten=not SAVE_CHUNK,
                       # ),
                       # ColOut(
                       #     "JetGoodFromHiggsOrdered",
                       #     [
                       #         "pt",
                       #         "eta",
                       #         "phi",
                       #         "mass",
                       #         "btagPNetB",
                       #     ],
                        #flatten=not SAVE_CHUNK,
                       # ),
                       #  ColOut(
                       #      "HH",
                       #      [
                       #          "pt",
                       #          "eta",
                       #          "phi",
                       #          "mass",
                       #          # "cos_theta_star",
                       #          # "dR",
                       #          # "dPhi",
                       #          # "dEta",
                       #      ],
                        #flatten=not SAVE_CHUNK,
                       #  ),
                        ColOut(
                            "events",
                            [
                                "best_pairing_probability",
                                "second_best_pairing_probability",
                                "Delta_pairing_probabilities",
                                # "random_pt_weights",
                                # "HT",
                                # "dR_min",
                                # "dR_max",
                            ],
                            flatten=not SAVE_CHUNK,
                        ),
                    ]
                    if CLASSIFICATION
                    else [
                        ColOut(
                            "events",
                            [
                                "random_pt_weights",
                            ],
                            flatten=not SAVE_CHUNK,
                        ),
                    ]
                    if RANDOM_PT
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
