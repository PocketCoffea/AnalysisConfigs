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
from workflow import QCDBaseProcessor

import custom_cut_functions
# import custom_cuts
from custom_cut_functions import *
# from custom_cuts import *
# from params.binning import bins
import os

localdir = os.path.dirname(os.path.abspath(__file__))

# Loading default parameters
from pocket_coffea.parameters import defaults

default_parameters = defaults.get_default_parameters()
defaults.register_configuration_dir("config_dir", localdir + "/params")

# Samples to exclude in specific histograms
# exclude_data = ["DATA_SingleEle", "DATA_SingleMuon"]
# exclude_nonttbar = ["ttHTobb", "TTTo2L2Nu", "SingleTop", "WJetsToLNu_HT"] + exclude_data

# adding object preselection
year = "2018"
parameters = defaults.merge_parameters_from_files(
    default_parameters,
    f"{localdir}/params/object_preselection.yaml",
    # f"{localdir}/params/triggers.yaml",
    # f"{localdir}/params/lepton_scale_factors.yaml",
    # f"{localdir}/params/plotting_style.yaml",
    update=True,
)

cfg = Configurator(
    parameters=parameters,
    datasets={
        "jsons": [
            f"{localdir}/datasets/QCD.json",
            f"{localdir}/datasets/DATA_SingleMuon.json",
        ],
        "filter": {
            "samples": [
                "QCD",
                "DATA_SingleMuon",

            ],
            "samples_exclude": [],
            "year": [year],
        },
        "subsamples": {
        },
    },
    workflow=QCDBaseProcessor,
    #workflow_options={"parton_jet_min_dR": 0.3},
    skim=[
        # get_nObj_min(4, 15.0, "Jet"),
        # get_HLTsel(primaryDatasets=["SingleEle", "SingleMuon"]),
    ],
    preselections=[],
    categories={
        "baseline": [passthrough],
        # "SingleEle_3b" : [ get_nElectron(1, coll="ElectronGood"), get_nObj_eq(3, coll="BJetGood") ],
    },
    weights={
        "common": {
            "inclusive": [
                # "genWeight",
                # "lumi",
                # "XS",

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
        **jet_hists(coll="JetGood", pos=0),
        # "GenJetMatched_pt" : HistConf(
        #     [Axis(coll="GenJetGoodMatched", field="pt", bins=100, start=0, stop=1000,
        #           label="$\Delta \eta_{bb}$")]
        # ),
        # "JetMatched_pt" : HistConf(
        #     [Axis(coll="JetMatched", field="pt", bins=100, start=0, stop=1000,
        #             label="jet_matched_pt")]
        # ),

        # "JetMatched_Response" : HistConf(
        #   [Axis(coll="JetMatched", field="Response", bins=100, start=0, stop=10, pos=None,
        #         label="Response")]
        # ),

        # **met_hists(coll="MET"),
        # "deltaRbb_min" : HistConf(
        #     [Axis(coll="events", field="deltaRbb_min", bins=50, start=0, stop=5,
        #           label="$\Delta R_{bb}$")]
        # ),
        # "deltaEtabb_min" : HistConf(
        #     [Axis(coll="events", field="deltaEtabb_min", bins=50, start=0, stop=5,
        #           label="$\Delta \eta_{bb}$")]
        # ),
        # "deltaPhibb_min" : HistConf(
        #     [Axis(coll="events", field="deltaPhibb_min", bins=50, start=0, stop=5,
        #           label="$\Delta \phi_{bb}$")]
        # ),
        # "mbb" : HistConf(
        #     [Axis(coll="events", field="mbb", bins=50, start=0, stop=500,
        #             label="$m_{bb}$ [GeV]")]
        # ),
        # "jets_Ht" : HistConf(
        #   [Axis(coll="events", field="JetGood_Ht", bins=100, start=0, stop=2500,
        #         label="Jets $H_T$ [GeV]")]
        # ),
        # "electron_etaSC_pt_leading" : HistConf(
        #     [
        #         Axis(coll="ElectronGood", field="pt", pos=0, type="variable",
        #              bins=bins["ElectronGood_pt"][year],
        #              label="Electron $p_{T}$ [GeV]",
        #              lim=(0,500)),
        #         Axis(coll="ElectronGood", field="etaSC", pos=0, type="variable",
        #              bins=bins["ElectronGood_etaSC"][year],
        #              label="Electron Supercluster $\eta$",
        #              lim=(-2.5,2.5)),
        #     ],
        #     exclude_categories=["SingleMuon_1b", "SingleMuon_2b", "SingleMuon_3b", "SingleMuon_4b"]
        # ),
    },
    columns={
        "common": {
            "inclusive": [],
        },
        "bysample": {
            # "QCD": {
            #     "bycategory": {
            #         "baseline": [
            #             ColOut("JetMatched", ["pt"]),
            #             ColOut("JetMatched", ["Response"]),
            #         ]
            #     }
            # },

            # "DATA_SingleMuon": {
            #     "bycategory": {
            #         "baseline": [
            #             ColOut("JetMatched", ["pt"]),
            #             ColOut("JetMatched", ["Response"]),
            #         ]
            #     }
            # },
        },

    },
)

run_options = {
        "executor"       : "dask/slurm",
        "env"            : "conda",
        "workers"        : 1,
        "scaleout"       : 50,
        "worker_image"   : "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/pocketcoffea:lxplus-cc7-latest",
        "queue"          : "standard",
        "walltime"       : "00:40:00",
        "mem_per_worker" : "4GB", # GB
        "disk_per_worker" : "1GB", # GB
        "exclusive"      : False,
        "chunk"          : 400000,
        "retries"        : 50,
        "treereduction"  : 20,
        "adapt"          : False
}


if "dask" in run_options["executor"]:
    import cloudpickle

    cloudpickle.register_pickle_by_value(workflow)
    cloudpickle.register_pickle_by_value(custom_cut_functions)
    # cloudpickle.register_pickle_by_value(custom_cuts)
