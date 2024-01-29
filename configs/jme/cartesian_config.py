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

from custom_cut_functions import *
from params.binning import *

import os

localdir = os.path.dirname(os.path.abspath(__file__))

# Loading default parameters
from pocket_coffea.parameters import defaults

default_parameters = defaults.get_default_parameters()
defaults.register_configuration_dir("config_dir", localdir + "/params")


# adding object preselection
year = "2018"
parameters = defaults.merge_parameters_from_files(
    default_parameters,
    f"{localdir}/params/object_preselection.yaml",
    update=True,
)


# cuts_pt = []
# cuts_names_pt = []
# for j in range(len(pt_bins) - 1):
#     pt_low, pt_high = pt_bins[j], pt_bins[j + 1]
#     cuts_pt.append(get_ptbin(pt_low, pt_high))
#     cuts_names_pt.append(f'pt{pt_low}to{pt_high}')

cuts_eta = []
cuts_names_eta = []
for i in range(len(eta_bins) - 1):
    eta_low, eta_high = eta_bins[i], eta_bins[i + 1]
    cuts_eta.append(get_etabin(eta_low, eta_high))
    cuts_names_eta.append(f"MatchedJets_eta{eta_low}to{eta_high}")


multicuts = [
    MultiCut(name="eta", cuts=cuts_eta, cuts_names=cuts_names_eta),
    # MultiCut(name="pt", #HERE
    #          cuts=cuts_pt,
    #          cuts_names=cuts_names_pt),
]

common_cats = {
    "baseline": [passthrough],
}


variables_dict = {
    **{
        f"MatchedJets{flav}_flav": HistConf(
            [
                Axis(
                    coll=f"MatchedJets{flav}",
                    field="partonFlavour",
                    bins=22,
                    start=0,
                    stop=22,
                    label=f"MatchedJets{flav}_flav",
                )
            ]
        )
        for flav in list([f"_{x}" for x in flav_dict.keys()]) + [""]
    },
    **{
        f"MatchedJets{flav}_JetPtJEC": HistConf(
            [
                Axis(
                    coll=f"MatchedJets{flav}",
                    field="JetPtJEC",
                    bins=100,
                    start=0,
                    stop=50,
                    label=f"MatchedJets{flav}_JetPtJEC",
                )
            ]
        )
        for flav in list([f"_{x}" for x in flav_dict.keys()]) + [""]
    },
    **{
        f"MatchedJets{flav}_JetPtRaw": HistConf(
            [
                Axis(
                    coll=f"MatchedJets{flav}",
                    field="JetPtRaw",
                    bins=100,
                    start=0,
                    stop=50,
                    label=f"MatchedJets{flav}_JetPtRaw",
                )
            ]
        )
        for flav in list([f"_{x}" for x in flav_dict.keys()]) + [""]
    },
    **{
        f"MatchedJets{flav}_GenJetPt": HistConf(
            [
                Axis(
                    coll=f"MatchedJets{flav}",
                    field="pt",
                    bins=100,
                    start=0,
                    stop=50,
                    label=f"MatchedJets{flav}_GenJetPt",
                )
            ]
        )
        for flav in list([f"_{x}" for x in flav_dict.keys()]) + [""]
    },
    **{
        f"MatchedJets{flav}_DeltaEta": HistConf(
            [
                Axis(
                    coll=f"MatchedJets{flav}",
                    field="DeltaEta",
                    bins=100,
                    start=-3,
                    stop=3,
                    label=f"MatchedJets{flav}_"+r"$\Delta \eta$",
                )
            ]
        )
        for flav in list([f"_{x}" for x in flav_dict.keys()]) + [""]
    },
    **{
        f"MatchedJets{flav}_DeltaPhi": HistConf(
            [
                Axis(
                    coll=f"MatchedJets{flav}",
                    field="DeltaPhi",
                    bins=100,
                    start=-3,
                    stop=3,
                    label=f"MatchedJets{flav}_"+r"$\Delta \phi$",
                )
            ]
        )
        for flav in list([f"_{x}" for x in flav_dict.keys()]) + [""]
    },
    **{
        f"MatchedJets{flav}_DeltaR": HistConf(
            [
                Axis(
                    coll=f"MatchedJets{flav}",
                    field="DeltaR",
                    bins=100,
                    start=0,
                    stop=3,
                    label=f"MatchedJets{flav}_"+r"$\Delta R$",
                )
            ]
        )
        for flav in list([f"_{x}" for x in flav_dict.keys()]) + [""]
    },
    **{
        f"MatchedJets{flav}_EtaRecoGen": HistConf(
            [
                Axis(
                    coll=f"MatchedJets{flav}",
                    field="EtaRecoGen",
                    bins=100,
                    start=-0.5,
                    stop=2,
                    label=f"MatchedJets{flav}_EtaRecoGen",
                )
            ]
        )
        for flav in list([f"_{x}" for x in flav_dict.keys()]) + [""]
    },
    # plot eta
    **{
        f"MatchedJets{flav}_eta": HistConf(
            [
                Axis(
                    coll=f"MatchedJets{flav}",
                    field="eta",
                    bins=100,
                    start=-5,
                    stop=5,
                    label=f"MatchedJets{flav}_eta",
                )
            ]
        )
        for flav in list([f"_{x}" for x in flav_dict.keys()]) + [""]
    },
    **{
        f"MatchedJets{flav}_ResponseJEC": HistConf(
            [
                Axis(
                    coll=f"MatchedJets{flav}",
                    field="ResponseJEC",
                    bins=100,
                    start=0,
                    stop=4,
                    pos=None,
                    label=f"MatchedJets{flav}_ResponseJEC",
                )
            ]
        )
        for flav in list([f"_{x}" for x in flav_dict.keys()]) + [""]
    },
    **{
        f"MatchedJets{flav}_ResponseRaw": HistConf(
            [
                Axis(
                    coll=f"MatchedJets{flav}",
                    field="ResponseRaw",
                    bins=100,
                    start=0,
                    stop=4,
                    pos=None,
                    label=f"MatchedJets{flav}_ResponseRaw",
                )
            ]
        )
        for flav in list([f"_{x}" for x in flav_dict.keys()]) + [""]
    },
    # # plot variables in eta bins and pt bins
    # **{
    #     f"MatchedJets_pt{pt_bins[j]}to{pt_bins[j+1]}_{var}": HistConf(
    #         [
    #             Axis(
    #                 coll=f"MatchedJets_pt{pt_bins[j]}to{pt_bins[j+1]}",
    #                 field=var,
    #                 bins=100,
    #                 start=start,
    #                 stop=stop,
    #                 label=f"MatchedJets_pt{pt_bins[j]}to{pt_bins[j+1]}_{var}",
    #             )
    #         ]
    #     )
    #     for j in range(len(pt_bins) - 1)  # for each pt bin
    #     for var, start, stop in zip(
    #         # ["Response"],
    #         # [4]
    #         ["pt", "eta", "Response"],
    #         [0, -5.5, 0],
    #         [100, 5.5, 4],
    #     )
    # },
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
}
if int(os.environ.get("PNET", 0)) == 1:
    variables_dict.update(
        {
            **{
                f"MatchedJets{flav}_ResponsePNetReg": HistConf(
                    [
                        Axis(
                            coll=f"MatchedJets{flav}",
                            field="ResponsePNetReg",
                            bins=100,
                            start=0,
                            stop=4,
                            pos=None,
                            label=f"MatchedJets{flav}_ResponsePNetReg",
                        )
                    ]
                )
                for flav in list([f"_{x}" for x in flav_dict.keys()]) + [""]
            },
            # **{
            #     f"JetMatched_PNetRegPtRawCorr": HistConf(
            #         [
            #             Axis(
            #                 coll=f"JetMatched",
            #                 field="PNetRegPtRawCorr",
            #                 bins=100,
            #                 start=0,
            #                 stop=4,
            #                 pos=None,
            #                 label="JetMatched_PNetRegPtRawCorr",
            #             )
            #         ]
            #     )
            # },
            # **{
            #     f"JetMatched_PNetRegPtRawCorrNeutrino": HistConf(
            #         [
            #             Axis(
            #                 coll=f"JetMatched",
            #                 field="PNetRegPtRawCorrNeutrino",
            #                 bins=100,
            #                 start=0,
            #                 stop=4,
            #                 pos=None,
            #                 label="JetMatched_PNetRegPtRawCorrNeutrino",
            #             )
            #         ]
            #     )
            # },
            # **{
            #     f"JetMatched_PNetRegPtRawCorrFull": HistConf(
            #         [
            #             Axis(
            #                 coll=f"JetMatched",
            #                 field="PNetRegPtRawCorrFull",
            #                 bins=100,
            #                 start=0,
            #                 stop=4,
            #                 pos=None,
            #                 label="JetMatched_PNetRegPtRawCorrFull",
            #             )
            #         ]
            #     )
            # },
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
                    ]
                )
                for flav in list([f"_{x}" for x in flav_dict.keys()]) + [""]
            },
            # **{
            #     f"MatchedJets{flav}_ResponsePNetRegNeutrinoVSpt": HistConf(
            #         [
            #             Axis(
            #                 coll=f"MatchedJets{flav}",
            #                 field="ResponsePNetRegNeutrino",
            #                 bins=response_bins,
            #                 pos=None,
            #                 label=f"MatchedJets{flav}_ResponsePNetRegNeutrino",
            #             ),
            #             Axis(
            #                 coll=f"MatchedJets{flav}",
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
            #     f"MatchedJets{flav}_ResponsePNetRegFullVSpt": HistConf(
            #         [
            #             Axis(
            #                 coll=f"MatchedJets{flav}",
            #                 field="ResponsePNetRegFull",
            #                 bins=response_bins,
            #                 pos=None,
            #                 label=f"MatchedJets{flav}_ResponsePNetRegFull",
            #             ),
            #             Axis(
            #                 coll=f"MatchedJets{flav}",
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


cfg = Configurator(
    parameters=parameters,
    datasets={
        "jsons": [
            f"{localdir}/datasets/QCD.json",
        ],
        "filter": {
            "samples": [
                "QCD_PT-15to7000_FlatPU"
                if int(os.environ.get("PNET", 0)) == 1
                else "QCD_PT-15to7000",
            ],
            "samples_exclude": [],
            "year": [year],
        },
        "subsamples": {},
    },
    workflow=QCDBaseProcessor,
    workflow_options={"donotscale_sumgenweights": True},
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
            # "inclusive": [
            #     ColOut("MatchedJets", ["Response", "pt", "eta"]),
            # ]
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
    "mem_per_worker": "10GB",  # 4GB
    "disk_per_worker": "1GB",
    "exclusive": False,
    "chunk": 400000,  # 400000
    "retries": 50,
    "treereduction": 20,  # 5,
    "adapt": False,
}


# if "dask" in run_options["executor"]:
#     import cloudpickle

#     cloudpickle.register_pickle_by_value(workflow)
#     cloudpickle.register_pickle_by_value(custom_cut_functions)
#     # cloudpickle.register_pickle_by_value(custom_cuts)
