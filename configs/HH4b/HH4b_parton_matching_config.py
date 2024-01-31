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
year = "2018"
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
        # get_nObj_min(4, 15.0, "Jet"),
        # get_HLTsel(primaryDatasets=["SingleEle", "SingleMuon"]),
    ],
    preselections=[hh4b_presel],
    categories={
        # "baseline": [passthrough],
        "hh4b_parton_matching": [get_nObj_eq(4, coll="PartonMatched")],
        # "SingleEle_3b" : [ get_nElectron(1, coll="ElectronGood"), get_nObj_eq(3, coll="BJetGood") ],
        # "SingleEle_>=4b" : [ get_nElectron(1, coll="ElectronGood"), get_nBtagMin(4, coll="BJetGood") ],
        # "SingleEle_>=5b" : [ get_nElectron(1, coll="ElectronGood"), get_nBtagMin(5, coll="BJetGood") ],
        # "SingleEle_>=6b" : [ get_nElectron(1, coll="ElectronGood"), get_nBtagMin(6, coll="BJetGood") ],
        # "SingleMuon_3b" : [ get_nMuon(1, coll="MuonGood"), get_nObj_eq(3, coll="BJetGood") ],
        # "SingleMuon_>=4b" : [ get_nMuon(1, coll="MuonGood"), get_nBtagMin(4, coll="BJetGood") ],
        # "SingleMuon_>=5b" : [ get_nMuon(1, coll="MuonGood"), get_nBtagMin(5, coll="BJetGood") ],
        # "SingleMuon_>=6b" : [ get_nMuon(1, coll="MuonGood"), get_nBtagMin(6, coll="BJetGood") ],
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
            "bysample": {}
            # },
            # "shape": {
            #     "common":{
            #         "inclusive": [ "JES_Total", "JER" ]
            #     }
        }
    },
    variables={
        **count_hist(coll="JetGood", bins=10, start=0, stop=10),
        **count_hist(coll="ElectronGood", bins=3, start=0, stop=3),
        **count_hist(coll="MuonGood", bins=3, start=0, stop=3),
        **count_hist(coll="JetGoodBtagOrderedMatched", bins=10, start=0, stop=10),
        **count_hist(coll="PartonMatched", bins=10, start=0, stop=10),
        **jet_hists(coll="JetGood", pos=0),
        **jet_hists(coll="JetGood", pos=1),
        **jet_hists(coll="JetGood", pos=2),
        **jet_hists(coll="JetGood", pos=3),
        **jet_hists(coll="JetGoodBtagOrdered", pos=0),
        **jet_hists(coll="JetGoodBtagOrdered", pos=1),
        **jet_hists(coll="JetGoodBtagOrdered", pos=2),
        **jet_hists(coll="JetGoodBtagOrdered", pos=3),
        **parton_hists(coll="PartonMatched", pos=0),
        **parton_hists(coll="PartonMatched", pos=1),
        **parton_hists(coll="PartonMatched", pos=2),
        **parton_hists(coll="PartonMatched", pos=3),
        **parton_hists(coll="PartonMatched"),
        **parton_hists(coll="JetGoodBtagOrderedMatched", pos=0),
        **parton_hists(coll="JetGoodBtagOrderedMatched", pos=1),
        **parton_hists(coll="JetGoodBtagOrderedMatched", pos=2),
        **parton_hists(coll="JetGoodBtagOrderedMatched", pos=3),
        **parton_hists(coll="JetGoodBtagOrderedMatched"),


    },
    columns={"common": {
        "inclusive": [
            ColOut("PartonMatched", ["provenance", "pdgId", "dRMatchedJet", "genPartIdxMother", "pt", "eta", "phi"]),
            ColOut("JetGoodBtagOrderedMatched", ["provenance", "pdgId", "dRMatchedJet", "pt", "eta", "phi", "btagPNetB"]),

        ],
        }, "bysample": {}},
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
