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
from pocket_coffea.lib.categorization import CartesianSelection, MultiCut

import workflow
from workflow import *

import custom_cut_functions
from cuts import *
from custom_functions import *

from custom_cut_functions import *
from params.binning import *

import os

localdir = os.path.dirname(os.path.abspath(__file__))

# Loading default parameters
from pocket_coffea.parameters import defaults

default_parameters = defaults.get_default_parameters()
defaults.register_configuration_dir("config_dir", localdir + "/params")


# adding object preselection
year = os.environ.get("YEAR", "2022_preEE")
# year = "2023_preBPix"
parameters = defaults.merge_parameters_from_files(
    default_parameters,
    f"{localdir}/params/object_preselection.yaml",
    f"{localdir}/params/jets_calibration.yaml",
    update=True,
)


mc_truth_corr_pnetreg = None
corr_files_pnetreg = {
    "2022_preEE": f"{localdir}/params/Summer22Run3_V3_MC_L2Relative_AK4PFPNet.txt",
    "2022_postEE": f"{localdir}/params/Summer22EERun3_V3_MC_L2Relative_AK4PFPNet.txt",
    "2023_preBPix": f"{localdir}/params/Summer23Run3_V3_MC_L2Relative_AK4PFPNet.txt",
    "2023_postBPix": f"{localdir}/params/Summer23BPixRun3_V3_MC_L2Relative_AK4PFPNet.txt",
}
if int(os.environ.get("CLOSURE", 0)) == 1:
    print(f"Performing closure test with {corr_files_pnetreg[year]}")
    mc_truth_corr_pnetreg = get_closure_function_information(corr_files_pnetreg[year])

mc_truth_corr_pnetreg_neutrino = None
corr_files_pnetreg_neutrino = {
    "2022_preEE": f"{localdir}/params/Summer22Run3_V3_MC_L2Relative_AK4PFPNetPlusNeutrino.txt",
    "2022_postEE": f"{localdir}/params/Summer22EERun3_V3_MC_L2Relative_AK4PFPNetPlusNeutrino.txt",
    "2023_preBPix": f"{localdir}/params/Summer23Run3_V3_MC_L2Relative_AK4PFPNetPlusNeutrino.txt",
    "2023_postBPix": f"{localdir}/params/Summer23BPixRun3_V3_MC_L2Relative_AK4PFPNetPlusNeutrino.txt",
}
if int(os.environ.get("CLOSURE", 0)) == 1:
    print(f"Performing closure test with {corr_files_pnetreg_neutrino[year]}")
    mc_truth_corr_pnetreg_neutrino = get_closure_function_information(
        corr_files_pnetreg_neutrino[year]
    )

mc_truth_corr = None
corr_files = {
    "2022_preEE": f"{localdir}/params/Summer22Run3_V1_MC_L2Relative_AK4PUPPI.txt",
    "2022_postEE": f"{localdir}/params/Summer22EEVetoRun3_V1_MC_L2Relative_AK4PUPPI.txt",
    "2023_preBPix": f"{localdir}/params/Summer23Run3_V1_MC_L2Relative_AK4PUPPI.txt",
    "2023_postBPix": f"{localdir}/params/Summer23BPixRun3_V3_MC_L2Relative_AK4PUPPI.txt",
}
print(f"Reapplying correctios {corr_files[year]}")
mc_truth_corr = get_closure_function_information(corr_files[year])

cuts_eta = []
cuts_names_eta = []
cuts_eta_neutrino = []
cuts_names_eta_neutrino = []
cuts_reco_eta = []
cuts_names_reco_eta = []

if int(os.environ.get("NEUTRINO", 1)) == 0:
    print("RECO JET ETA CUTS NEUTRINO==0")
    for i in range(len(eta_bins) - 1):
        eta_low, eta_high = eta_bins[i], eta_bins[i + 1]
        cuts_reco_eta.append(get_reco_etabin(eta_low, eta_high))
        cuts_names_reco_eta.append(f"MatchedJets_reco_eta{eta_low}to{eta_high}")
elif int(os.environ.get("NEUTRINO", 0)) == 1:
    print("RECO JET ETA CUTS NEUTRINO==1")
    for i in range(len(eta_bins) - 1):
        eta_low, eta_high = eta_bins[i], eta_bins[i + 1]
        cuts_reco_eta.append(get_reco_neutrino_etabin(eta_low, eta_high))
        cuts_names_reco_eta.append(f"MatchedJetsNeutrino_reco_eta{eta_low}to{eta_high}")
elif int(os.environ.get("ABS_ETA_INCLUSIVE", 0)) == 1:
    print("RUNNING ABS_ETA_INCLUSIVE ETA BINS")
    for i in range(len(eta_bins) - 1):
        eta_low, eta_high = eta_bins[i], eta_bins[i + 1]
        cuts_reco_eta.append(get_reco_neutrino_abs_etabin(eta_low, eta_high))
        cuts_names_reco_eta.append(
            f"MatchedJetsNeutrino_reco_abseta{eta_low}to{eta_high}"
        )
else:
    print("RECO JET ETA CUTS")
    for i in range(len(eta_bins) - 1):
        eta_low, eta_high = eta_bins[i], eta_bins[i + 1]
        cuts_reco_eta.append(get_reco_neutrino_etabin(eta_low, eta_high))
        cuts_names_reco_eta.append(f"MatchedJetsNeutrino_reco_eta{eta_low}to{eta_high}")

multicuts = [
    MultiCut(
        name="eta",
        cuts=cuts_eta + cuts_eta_neutrino + cuts_reco_eta,
        cuts_names=cuts_names_eta + cuts_names_eta_neutrino + cuts_names_reco_eta,
    ),
]

common_cats = {
    "baseline": [passthrough],
}

variables_dict = (
    {
        **{
            f"MatchedJets{flav}_ResponseJECVSpt": HistConf(
                [
                    Axis(
                        coll=f"MatchedJets{flav}",
                        field="ResponseJEC",
                        bins=response_bins,
                        pos=None,
                        label=f"MatchedJets{flav}_ResponseJEC",
                    ),
                    Axis(
                        coll=f"MatchedJets{flav}",
                        field="pt",
                        bins=pt_bins,
                        label=f"MatchedJets{flav}_pt",
                        type="variable",
                        pos=None,
                    ),
                ]
            )
            for flav in list([f"_{x}" for x in flav_dict.keys()]) + [""]
        },
        **{
            f"MatchedJets{flav}_JetPtJECVSpt": HistConf(
                [
                    Axis(
                        coll=f"MatchedJets{flav}",
                        field="JetPtJEC",
                        bins=jet_pt_bins,
                        pos=None,
                        label=f"MatchedJets{flav}_JetPtJEC",
                    ),
                    Axis(
                        coll=f"MatchedJets{flav}",
                        field="pt",
                        bins=pt_bins,
                        label=f"MatchedJets{flav}_pt",
                        type="variable",
                        pos=None,
                    ),
                ]
            )
            for flav in list([f"_{x}" for x in flav_dict.keys()]) + [""]
        },
        **{
            f"MatchedJets{flav}_ResponseRawVSpt": HistConf(
                [
                    Axis(
                        coll=f"MatchedJets{flav}",
                        field="ResponseRaw",
                        bins=response_bins,
                        pos=None,
                        label=f"MatchedJets{flav}_ResponseRaw",
                    ),
                    Axis(
                        coll=f"MatchedJets{flav}",
                        field="pt",
                        bins=pt_bins,
                        label=f"MatchedJets{flav}_pt",
                        type="variable",
                        pos=None,
                    ),
                ]
            )
            for flav in list([f"_{x}" for x in flav_dict.keys()]) + [""]
        },
        **{
            f"MatchedJets{flav}_JetPtRawVSpt": HistConf(
                [
                    Axis(
                        coll=f"MatchedJets{flav}",
                        field="JetPtRaw",
                        bins=jet_pt_bins,
                        pos=None,
                        label=f"MatchedJets{flav}_JetPtRaw",
                    ),
                    Axis(
                        coll=f"MatchedJets{flav}",
                        field="pt",
                        bins=pt_bins,
                        label=f"MatchedJets{flav}_pt",
                        type="variable",
                        pos=None,
                    ),
                ]
            )
            for flav in list([f"_{x}" for x in flav_dict.keys()]) + [""]
        },
    }
    if int(os.environ.get("NEUTRINO", 0)) == 0
    else {}
)
if int(os.environ.get("PNET", 0)) == 1 and int(os.environ.get("NEUTRINO", 0)) == 0:
    variables_dict.update(
        {
            **{
                f"MatchedJets{flav}_ResponsePNetRegVSpt": HistConf(
                    [
                        Axis(
                            coll=f"MatchedJets{flav}",
                            field="ResponsePNetReg",
                            bins=response_bins,
                            pos=None,
                            label=f"MatchedJets{flav}_ResponsePNetReg",
                        ),
                        Axis(
                            coll=f"MatchedJets{flav}",
                            field="pt",
                            bins=pt_bins,
                            label=f"MatchedJets{flav}_pt",
                            type="variable",
                            pos=None,
                        ),
                    ],
                    only_categories=cuts_names_eta + cuts_names_reco_eta,
                )
                for flav in list([f"_{x}" for x in flav_dict.keys()]) + [""]
            },
            **{
                f"MatchedJets{flav}_JetPtPNetRegVSpt": HistConf(
                    [
                        Axis(
                            coll=f"MatchedJets{flav}",
                            field="JetPtPNetReg",
                            bins=jet_pt_bins,
                            pos=None,
                            label=f"MatchedJets{flav}_JetPtPNetReg",
                        ),
                        Axis(
                            coll=f"MatchedJets{flav}",
                            field="pt",
                            bins=pt_bins,
                            label=f"MatchedJets{flav}_pt",
                            type="variable",
                            pos=None,
                        ),
                    ],
                    only_categories=cuts_names_eta + cuts_names_reco_eta,
                )
                for flav in list([f"_{x}" for x in flav_dict.keys()]) + [""]
            },
        }
    )
    if int(os.environ.get("SPLITPNETREG15", 0)) == 1:
        variables_dict.update(
            {
                **{
                    f"MatchedJets{flav}_ResponsePNetRegSplit15VSpt": HistConf(
                        [
                            Axis(
                                coll=f"MatchedJetsSplit15{flav}",
                                field="ResponsePNetReg",
                                bins=response_bins,
                                pos=None,
                                label=f"MatchedJets{flav}_ResponsePNetReg",
                            ),
                            Axis(
                                coll=f"MatchedJetsSplit15{flav}",
                                field="pt",
                                bins=pt_bins,
                                label=f"MatchedJets{flav}_pt",
                                type="variable",
                                pos=None,
                            ),
                        ],
                        only_categories=cuts_names_eta + cuts_names_reco_eta,
                    )
                    for flav in list([f"_{x}" for x in flav_dict.keys()]) + [""]
                },
                **{
                    f"MatchedJets{flav}_JetPtPNetRegSplit15VSpt": HistConf(
                        [
                            Axis(
                                coll=f"MatchedJetsSplit15{flav}",
                                field="JetPtPNetReg",
                                bins=jet_pt_bins,
                                pos=None,
                                label=f"MatchedJets{flav}_JetPtPNetReg",
                            ),
                            Axis(
                                coll=f"MatchedJetsSplit15{flav}",
                                field="pt",
                                bins=pt_bins,
                                label=f"MatchedJets{flav}_pt",
                                type="variable",
                                pos=None,
                            ),
                        ],
                        only_categories=cuts_names_eta + cuts_names_reco_eta,
                    )
                    for flav in list([f"_{x}" for x in flav_dict.keys()]) + [""]
                },
            }
        )

if int(os.environ.get("PNET", 0)) == 1 and int(os.environ.get("NEUTRINO", 1)) == 1:
    variables_dict.update(
        {
            **{
                f"MatchedJets{flav}_ResponsePNetRegNeutrinoVSpt": HistConf(
                    [
                        Axis(
                            coll=f"MatchedJetsNeutrino{flav}",
                            field="ResponsePNetRegNeutrino",
                            bins=response_bins,
                            pos=None,
                            label=f"MatchedJets{flav}_ResponsePNetRegNeutrino",
                        ),
                        Axis(
                            coll=f"MatchedJetsNeutrino{flav}",
                            field="pt",
                            bins=pt_bins,
                            label=f"MatchedJets{flav}_pt",
                            type="variable",
                            pos=None,
                        ),
                    ],
                    only_categories=cuts_names_eta_neutrino + cuts_names_reco_eta,
                )
                for flav in list([f"_{x}" for x in flav_dict.keys()]) + [""]
            },
            **{
                f"MatchedJets{flav}_JetPtPNetRegNeutrinoVSpt": HistConf(
                    [
                        Axis(
                            coll=f"MatchedJetsNeutrino{flav}",
                            field="JetPtPNetRegNeutrino",
                            bins=jet_pt_bins,
                            pos=None,
                            label=f"MatchedJets{flav}_JetPtPNetRegNeutrino",
                        ),
                        Axis(
                            coll=f"MatchedJetsNeutrino{flav}",
                            field="pt",
                            bins=pt_bins,
                            label=f"MatchedJets{flav}_pt",
                            type="variable",
                            pos=None,
                        ),
                    ],
                    only_categories=cuts_names_eta_neutrino + cuts_names_reco_eta,
                )
                for flav in list([f"_{x}" for x in flav_dict.keys()]) + [""]
            },
        }
    )
    if int(os.environ.get("SPLITPNETREG15", 0)) == 1:
        variables_dict.update(
            {
                **{
                    f"MatchedJets{flav}_ResponsePNetRegNeutrinoSplit15VSpt": HistConf(
                        [
                            Axis(
                                coll=f"MatchedJetsNeutrinoSplit15{flav}",
                                field="ResponsePNetRegNeutrino",
                                bins=response_bins,
                                pos=None,
                                label=f"MatchedJets{flav}_ResponsePNetRegNeutrino",
                            ),
                            Axis(
                                coll=f"MatchedJetsNeutrinoSplit15{flav}",
                                field="pt",
                                bins=pt_bins,
                                label=f"MatchedJets{flav}_pt",
                                type="variable",
                                pos=None,
                            ),
                        ],
                        only_categories=cuts_names_eta_neutrino + cuts_names_reco_eta,
                    )
                    for flav in list([f"_{x}" for x in flav_dict.keys()]) + [""]
                },
                **{
                    f"MatchedJets{flav}_JetPtPNetRegNeutrinoSplit15VSpt": HistConf(
                        [
                            Axis(
                                coll=f"MatchedJetsNeutrinoSplit15{flav}",
                                field="JetPtPNetRegNeutrino",
                                bins=jet_pt_bins,
                                pos=None,
                                label=f"MatchedJets{flav}_JetPtPNetRegNeutrino",
                            ),
                            Axis(
                                coll=f"MatchedJetsNeutrinoSplit15{flav}",
                                field="pt",
                                bins=pt_bins,
                                label=f"MatchedJets{flav}_pt",
                                type="variable",
                                pos=None,
                            ),
                        ],
                        only_categories=cuts_names_eta_neutrino + cuts_names_reco_eta,
                    )
                    for flav in list([f"_{x}" for x in flav_dict.keys()]) + [""]
                },
            }
        )
if int(os.environ.get("NEUTRINO", 1)) == 1 and int(os.environ.get("CLOSURE", 0)) == 1:
    variables_dict.update(
        {
            **{
                f"MatchedJets{flav}_ResponseJECNeutrinoVSpt": HistConf(
                    [
                        Axis(
                            coll=f"MatchedJetsNeutrino{flav}",
                            field="ResponseJEC",
                            bins=response_bins,
                            pos=None,
                            label=f"MatchedJets{flav}_ResponseJECNeutrino",
                        ),
                        Axis(
                            coll=f"MatchedJetsNeutrino{flav}",
                            field="pt",
                            bins=pt_bins,
                            label=f"MatchedJets{flav}_pt",
                            type="variable",
                            pos=None,
                        ),
                    ]
                )
                for flav in list([f"_{x}" for x in flav_dict.keys()]) + [""]
            },
            **{
                f"MatchedJets{flav}_JetPtJECNeutrinoVSpt": HistConf(
                    [
                        Axis(
                            coll=f"MatchedJetsNeutrino{flav}",
                            field="JetPtJEC",
                            bins=jet_pt_bins,
                            pos=None,
                            label=f"MatchedJets{flav}_JetPtJECNeutrino",
                        ),
                        Axis(
                            coll=f"MatchedJetsNeutrino{flav}",
                            field="pt",
                            bins=pt_bins,
                            label=f"MatchedJets{flav}_pt",
                            type="variable",
                            pos=None,
                        ),
                    ]
                )
                for flav in list([f"_{x}" for x in flav_dict.keys()]) + [""]
            },
            # **{
            #     f"MatchedJets{flav}_ResponseRawNeutrinoVSpt": HistConf(
            #         [
            #             Axis(
            #                 coll=f"MatchedJetsNeutrino{flav}",
            #                 field="ResponseRaw",
            #                 bins=response_bins,
            #                 pos=None,
            #                 label=f"MatchedJets{flav}_ResponseRawNeutrino",
            #             ),
            #             Axis(
            #                 coll=f"MatchedJetsNeutrino{flav}",
            #                 field="pt",
            #                 bins=pt_bins,
            #                 label=f"MatchedJets{flav}_pt",
            #                 type="variable",
            #                 pos=None,
            #             ),
            #         ]
            #     )
            #     for flav in list([f"_{x}" for x in flav_dict.keys()]) + [""]
            # },
            # **{
            #     f"MatchedJets{flav}_JetPtRawNeutrinoVSpt": HistConf(
            #         [
            #             Axis(
            #                 coll=f"MatchedJetsNeutrino{flav}",
            #                 field="JetPtRaw",
            #                 bins=jet_pt_bins,
            #                 pos=None,
            #                 label=f"MatchedJets{flav}_JetPtRawNeutrino",
            #             ),
            #             Axis(
            #                 coll=f"MatchedJetsNeutrino{flav}",
            #                 field="pt",
            #                 bins=pt_bins,
            #                 label=f"MatchedJets{flav}_pt",
            #                 type="variable",
            #                 pos=None,
            #             ),
            #         ]
            #     )
            #     for flav in list([f"_{x}" for x in flav_dict.keys()]) + [""]
            # },
        }
    )

# if int(os.environ.get("CLOSURE", 0)) == 1:
if False:
    variables_dict.update(
        {
            **{
                f"MatchedJets{flav}_MCTruthCorrPNetRegVSJetPtPNetReg": HistConf(
                    [
                        Axis(
                            coll=f"MatchedJets{flav}",
                            field="MCTruthCorrPNetReg",
                            bins=list(np.linspace(0, 2, 1000)),
                            pos=None,
                            label=f"MatchedJets{flav}_MCTruthCorrPNetReg",
                        ),
                        Axis(
                            coll=f"MatchedJets{flav}",
                            field="JetPtPNetReg",
                            bins=pt_bins,
                            label=f"MatchedJets{flav}_JetPtPNetReg",
                            type="variable",
                            pos=None,
                        ),
                    ],
                    only_categories=cuts_names_eta_neutrino + cuts_names_reco_eta,
                )
                for flav in list([f"_{x}" for x in flav_dict.keys()]) + [""]
            },
            **{
                f"MatchedJets{flav}_MCTruthCorrPNetRegNeutrinoVSJetPtPNetRegNeutrino": HistConf(
                    [
                        Axis(
                            coll=f"MatchedJetsNeutrino{flav}",
                            field="MCTruthCorrPNetRegNeutrino",
                            bins=list(np.linspace(0, 2, 1000)),
                            pos=None,
                            label=f"MatchedJets{flav}_MCTruthCorrPNetRegNeutrino",
                        ),
                        Axis(
                            coll=f"MatchedJetsNeutrino{flav}",
                            field="JetPtPNetRegNeutrino",
                            bins=pt_bins,
                            label=f"MatchedJets{flav}_JetPtPNetRegNeutrino",
                            type="variable",
                            pos=None,
                        ),
                    ],
                    only_categories=cuts_names_eta_neutrino + cuts_names_reco_eta,
                )
                for flav in list([f"_{x}" for x in flav_dict.keys()]) + [""]
            },
        }
    )


samples_dict = {
    "2022_preEE": "QCD_PT-15to7000_JMENano_Summer22",
    "2022_postEE": "QCD_PT-15to7000_JMENano_Summer22EE",
    "2023_preBPix": "QCD_PT-15to7000_JMENano_Summer23",
    "2023_postBPix": "QCD_PT-15to7000_JMENano_Summer23BPix",
}
samples_PNetReg15_dict = {
    "2022_preEE": "QCD_PT-15to7000_PNetReg15_JMENano_Summer22",
    "2022_postEE": "QCD_PT-15to7000_PNetReg15_JMENano_Summer22EE",
    "2023_preBPix": "QCD_PT-15to7000_PNetReg15_JMENano_Summer23",
    "2023_postBPix": "QCD_PT-15to7000_PNetReg15_JMENano_Summer23BPix",
}

cfg = Configurator(
    parameters=parameters,
    datasets={
        "jsons": [
            # f"{localdir}/datasets/QCD.json",
            f"{localdir}/datasets/QCD_redirector.json",
            f"{localdir}/datasets/QCD_PNetReg15.json",
        ],
        "filter": {
            "samples": [
                (
                    samples_PNetReg15_dict[year]
                    if (
                        int(os.environ.get("PNETREG15", 0)) == 1
                        or int(os.environ.get("SPLITPNETREG15", 0)) == 1
                    )
                    else samples_dict[year]
                )
            ],
            "samples_exclude": [],
            "year": [year],
        },
        "subsamples": {},
    },
    workflow=QCDBaseProcessor,
    workflow_options={
        "donotscale_sumgenweights": True,
        "mc_truth_corr_pnetreg": mc_truth_corr_pnetreg,
        "mc_truth_corr_pnetreg_neutrino": mc_truth_corr_pnetreg_neutrino,
        "mc_truth_corr": mc_truth_corr,
        "DeltaR_matching": 0.2,
        "SetRegResponseToZero": True,
        "GenJetPtCut": (
            15
            if (
                int(os.environ.get("PNETREG15", 0)) == 1
                or int(os.environ.get("SPLITPNETREG15", 0)) == 1
            )
            else 50
        ),
    },
    skim=[],
    preselections=[],
    categories=CartesianSelection(multicuts=multicuts, common_cats=common_cats),
    # categories={
    #             **common_cats,
    #     # **eta_cuts,
    # },
    weights={
        "common": {
            "inclusive": [
                # "genWeight",
                # "lumi",
                # "XS",
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
    variables=variables_dict,
    columns={
        "common": {
            "inclusive": [
                # ColOut("MatchedJets", [ "pt", "eta"]),
                # ColOut("MatchedJetsNeutrino_reshape", [ "pt", "eta"]),
            ]
            # + [
            #     ColOut(
            #         f"MatchedJets_pt{pt_bins[j]}to{pt_bins[j+1]}",
            #         ["Response", "pt", "eta"],
            #         # fill_none=False,
            #     )
            #     for j in range(len(pt_bins) - 1)  # for each pt bin
            # ],
            # cat: [ColOut("MatchedJets", ["Response", "pt", "eta"])]
            # + [
            #     ColOut(
            #         f"MatchedJets_pt{pt_bins[j]}to{pt_bins[j+1]}",
            #         ["Response", "pt", "eta"],
            #         # fill_none=False,
            #     )
            #     for j in range(len(pt_bins) - 1)  # for each pt bin
            # ]
            # for cat in cuts_names_eta
        },
        "bysample": {
            # "QCD": {
            #     "bycategory": {
            #         # "baseline": [
            #         #     ColOut("MatchedJets", ["pt", "eta"]),
            #         #     # ColOut("MatchedJets", ["Response", "pt", "eta"]),
            #         # ],
            #         # + [
            #         #     ColOut(
            #         #         f"MatchedJets_pt{pt_bins[j]}to{pt_bins[j+1]}",
            #         #         ["Response", "pt", "eta"],
            #         #         # fill_none=False,
            #         #     )
            #         #     for j in range(len(pt_bins) - 1)  # for each pt bin
            #         # ],
            #         cat: [
            #             ColOut(
            #                 f"MatchedJets_pt{pt_bins[j]}to{pt_bins[j+1]}",
            #                 ["Response", "pt", "eta"],
            #             )
            #             for j in range(len(pt_bins) - 1)  # for each pt bin
            #         ]
            #         for cat in cuts_names_eta
            #     }
            # },
        },
    },
)

run_options = {
    "executor": "dask/slurm",
    "env": "conda",
    "workers": 1,
    "scaleout": 200,
    "worker_image": "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/pocketcoffea:lxplus-cc7-latest",
    "queue": "standard",
    "walltime": "00:40:00",  # 00:40:00
    "mem_per_worker": "8GB",  # 4GB
    "disk_per_worker": "1GB",
    "exclusive": False,
    "chunk": 100000,  # 400000
    "retries": 50,
    "treereduction": 20,  # 5,
    "adapt": False,
}


if "dask" in run_options["executor"]:
    import cloudpickle

    cloudpickle.register_pickle_by_value(workflow)
    cloudpickle.register_pickle_by_value(custom_cut_functions)
    # cloudpickle.register_pickle_by_value(custom_cuts)
