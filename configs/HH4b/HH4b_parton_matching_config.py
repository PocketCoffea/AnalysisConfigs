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
from workflow import HH4bPartonMatchingProcessor

import custom_cut_functions
import custom_cuts
from custom_cut_functions import *
from custom_cuts import *

# from params.binning import bins
import os

localdir = os.path.dirname(os.path.abspath(__file__))

# Loading default parameters
from pocket_coffea.parameters import defaults

default_parameters = defaults.get_default_parameters()
defaults.register_configuration_dir("config_dir", localdir + "/params")

# adding object preselection
year = "2018"  # TODO: change year to 2022
parameters = defaults.merge_parameters_from_files(
    default_parameters,
    f"{localdir}/params/object_preselection.yaml",
    f"{localdir}/params/triggers.yaml",
    # f"{localdir}/params/lepton_scale_factors.yaml",
    # f"{localdir}/params/plotting_style.yaml",
    update=True,
)

cfg = Configurator(
    parameters=parameters,
    datasets={
        "jsons": [
            # f"{localdir}/datasets/DATA_JetMET.json"
            f"{localdir}/datasets/signal_ggF_HH4b.json",
        ],
        "filter": {
            "samples": [
                "GluGlutoHHto4B",
                # "DATA_JetMET",
            ],
            "samples_exclude": [],
            "year": [year],
        },
        "subsamples": {},
    },
    workflow=HH4bPartonMatchingProcessor,
    workflow_options={"parton_jet_min_dR": 0.4, "max_num_jets": 4},
    skim=[
        get_HLTsel(primaryDatasets=["JetMET"]),
    ],
    preselections=[
        # lepton_veto_presel,
        # four_jet_presel,
        # jet_pt_presel,
        # jet_btag_presel,
        # hh4b_presel,
    ],
    categories={
        "lepton_veto": [lepton_veto_presel],
        "four_jet": [four_jet_presel],
        "jet_pt": [jet_pt_presel],
        "jet_btag": [jet_btag_presel],
        "jet_pt_copy": [jet_pt_presel],
        "jet_btag_loose": [jet_btag_loose_presel],
        "baseline": [jet_btag_presel],
        "full_parton_matching": [jet_btag_presel, get_nObj_eq(4, coll="PartonMatched")],
    },
    weights={
        "common": {
            "inclusive": [
                "genWeight",
                "lumi",
                "XS",
                # "pileup",
                # "sf_ele_reco",
                # "sf_ele_id",
                # "sf_ele_trigger",
                # "sf_mu_id",
                # "sf_mu_iso",
                # "sf_mu_trigger",
                # "sf_btag",  # "sf_btag_calib",
                # "sf_jet_puId",
            ],
            "bycategory": {},
        },
        "bysample": {},
    },
    variations={
        "weights": {
            "common": {
                "inclusive": [
                    # "pileup",
                    # "sf_ele_reco",
                    # "sf_ele_id",
                    # "sf_mu_id",
                    # "sf_mu_iso",
                    # "sf_mu_trigger",
                    # "sf_jet_puId",
                ],  # + [ f"sf_btag_{b}" for b in parameters["systematic_variations"]["weight_variations"]["sf_btag"][year]],
                # + [f"sf_ele_trigger_{v}" for v in parameters["systematic_variations"]["weight_variations"]["sf_ele_trigger"][year]],
                "bycategory": {},
            },
            "bysample": {},
            # },
            # "shape": {
            #     "common":{
            #         "inclusive": [ "JES_Total", "JER" ]
            #     }
        }
    },
    variables={
        **count_hist(coll="JetGood", bins=10, start=0, stop=10),
        **count_hist(coll="JetGoodBTagOrder", bins=10, start=0, stop=10),
        **count_hist(coll="ElectronGood", bins=3, start=0, stop=3),
        **count_hist(coll="MuonGood", bins=3, start=0, stop=3),
        **count_hist(coll="JetGoodBTagOrderMatched", bins=10, start=0, stop=10),
        **count_hist(coll="PartonMatched", bins=10, start=0, stop=10),
        **jet_hists(coll="JetGood", pos=0),
        **jet_hists(coll="JetGood", pos=1),
        **jet_hists(coll="JetGood", pos=2),
        **jet_hists(coll="JetGood", pos=3),
        **jet_hists(coll="JetGoodPtOrder", pos=0),
        **jet_hists(coll="JetGoodPtOrder", pos=1),
        **jet_hists(coll="JetGoodPtOrder", pos=2),
        **jet_hists(coll="JetGoodPtOrder", pos=3),
        **jet_hists(coll="JetGoodBTagOrder", pos=0),
        **jet_hists(coll="JetGoodBTagOrder", pos=1),
        **jet_hists(coll="JetGoodBTagOrder", pos=2),
        **jet_hists(coll="JetGoodBTagOrder", pos=3),
        **parton_hists(coll="PartonMatched", pos=0),
        **parton_hists(coll="PartonMatched", pos=1),
        **parton_hists(coll="PartonMatched", pos=2),
        **parton_hists(coll="PartonMatched", pos=3),
        **parton_hists(coll="PartonMatched"),
        **parton_hists(coll="JetGoodBTagOrderMatched", pos=0),
        **parton_hists(coll="JetGoodBTagOrderMatched", pos=1),
        **parton_hists(coll="JetGoodBTagOrderMatched", pos=2),
        **parton_hists(coll="JetGoodBTagOrderMatched", pos=3),
        **parton_hists(coll="JetGoodBTagOrderMatched"),
        # **{
        #     f"GenHiggs1Mass": HistConf(
        #         [
        #             Axis(
        #                 coll=f"events",
        #                 field="GenHiggs1Mass",
        #                 bins=60,
        #                 start=123,
        #                 stop=126,
        #                 label=f"GenHiggs1Mass",
        #             )
        #         ]
        #     )
        # },
        # **{
        #     f"GenHiggs2Mass": HistConf(
        #         [
        #             Axis(
        #                 coll=f"events",
        #                 field="GenHiggs2Mass",
        #                 bins=60,
        #                 start=123,
        #                 stop=126,
        #                 label=f"GenHiggs2Mass",
        #             )
        #         ]
        #     )
        # },
        **{
            f"RecoHiggs1Mass": HistConf(
                [
                    Axis(
                        coll=f"events",
                        field="RecoHiggs1Mass",
                        bins=30,
                        start=60,
                        stop=200,
                        label=f"RecoHiggs1Mass",
                    )
                ]
            )
        },
        **{
            f"RecoHiggs2Mass": HistConf(
                [
                    Axis(
                        coll=f"events",
                        field="RecoHiggs2Mass",
                        bins=30,
                        start=60,
                        stop=200,
                        label=f"RecoHiggs2Mass",
                    )
                ]
            )
        },
        **{
            f"PNetRegRecoHiggs1Mass": HistConf(
                [
                    Axis(
                        coll=f"events",
                        field="PNetRegRecoHiggs1Mass",
                        bins=30,
                        start=60,
                        stop=200,
                        label=f"PNetRegRecoHiggs1Mass",
                    )
                ]
            )
        },
        **{
            f"PNetRegRecoHiggs2Mass": HistConf(
                [
                    Axis(
                        coll=f"events",
                        field="PNetRegRecoHiggs2Mass",
                        bins=30,
                        start=60,
                        stop=200,
                        label=f"PNetRegRecoHiggs2Mass",
                    )
                ]
            )
        },
        # **{
        #     f"AllGenHiggs1Mass": HistConf(
        #         [
        #             Axis(
        #                 coll=f"events",
        #                 field="AllGenHiggs1Mass",
        #                 bins=80,
        #                 start=120,
        #                 stop=130,
        #                 label=f"AllGenHiggs1Mass",
        #             )
        #         ]
        #     )
        # },
        # **{
        #     f"AllGenHiggs2Mass": HistConf(
        #         [
        #             Axis(
        #                 coll=f"events",
        #                 field="AllGenHiggs2Mass",
        #                 bins=80,
        #                 start=120,
        #                 stop=130,
        #                 label=f"AllGenHiggs2Mass",
        #             )
        #         ]
        #     )
        # },
    },
    columns={
        "common": {
            "inclusive": [
                ColOut(
                    "PartonMatched",
                    [
                        "provenance",
                        "pdgId",
                        "dRMatchedJet",
                        "genPartIdxMother",
                        "pt",
                        "eta",
                        "phi",
                    ],
                ),
                ColOut(
                    "JetGoodBTagOrderMatched",
                    [
                        "provenance",
                        "pdgId",
                        "dRMatchedJet",
                        "pt",
                        "eta",
                        "phi",
                        "btagPNetB",
                        "ptPnetRegNeutrino",
                    ],
                ),
                ColOut(
                    "JetGoodBTagOrder",
                    ["pt", "eta", "phi", "btagPNetB", "ptPnetRegNeutrino"],
                ),
                ColOut(
                    "JetGood", ["pt", "eta", "phi", "btagPNetB", "ptPnetRegNeutrino"]
                ),
                ColOut(
                    "JetGoodPtOrder",
                    ["pt", "eta", "phi", "btagPNetB", "ptPnetRegNeutrino"],
                ),
                ColOut(
                    "events",
                    [
                        "GenHiggs1Mass",
                        "GenHiggs2Mass",
                        "RecoHiggs1Mass",
                        "RecoHiggs2Mass",
                        "PNetRegRecoHiggs1Mass",
                        "PNetRegRecoHiggs2Mass",
                        "PNetRegNeutrinoRecoHiggs1Mass",
                        "PNetRegNeutrinoRecoHiggs2Mass",
                        "GenHiggs1Pt",
                        "GenHiggs2Pt",
                        "RecoHiggs1Pt",
                        "RecoHiggs2Pt",
                        "PNetRegRecoHiggs1Pt",
                        "PNetRegRecoHiggs2Pt",
                        "PNetRegNeutrinoRecoHiggs1Pt",
                        "PNetRegNeutrinoRecoHiggs2Pt",
                    ],
                ),
            ],
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
    "walltime": "00:40:00",
    "mem_per_worker": "6GB",  # GB
    "disk_per_worker": "1GB",  # GB
    "exclusive": False,
    "chunk": 400000,
    "retries": 50,
    "treereduction": 5,
    "adapt": False,
}


if "dask" in run_options["executor"]:
    import cloudpickle

    cloudpickle.register_pickle_by_value(workflow)
    cloudpickle.register_pickle_by_value(custom_cut_functions)
    cloudpickle.register_pickle_by_value(custom_cuts)
