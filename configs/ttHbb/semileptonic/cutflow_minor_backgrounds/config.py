from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.cut_definition import Cut
from pocket_coffea.lib.columns_manager import ColOut
from pocket_coffea.lib.cut_functions import get_nObj_min, get_HLTsel, get_nBtagMin, goldenJson, eventFlags, get_nPVgood
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.parameters.histograms import *
from pocket_coffea.workflows.tthbb_base_processor import ttHbbBaseProcessor


import custom_cut_functions
import custom_cuts
from custom_cut_functions import *
from custom_cuts import *
import os
localdir = os.path.dirname(os.path.abspath(__file__))

# Loading default parameters
from pocket_coffea.parameters import defaults
default_parameters = defaults.get_default_parameters()
defaults.register_configuration_dir("config_dir", localdir+"/params")

parameters = defaults.merge_parameters_from_files(default_parameters,
                                                  f"{localdir}/params/object_preselection.yaml",
                                                  f"{localdir}/params/triggers.yaml",
                                                  update=True)

cfg = Configurator(
    parameters = parameters,
    datasets = {
        "jsons": [
                  f"{localdir}/datasets/TTTT_TuneCP5_13TeV-amcatnlo-pythia8.json",
                    f"{localdir}/datasets/TTTJ_TuneCP5_13TeV-madgraph-pythia8.json",
                    f"{localdir}/datasets/TWZ-Zto2Q_TuneCP5_13TeV_madgraph-pythia8.json",
                    f"{localdir}/datasets/TWZToLL_thad_Wlept_5f_DR_TuneCP5_13TeV-amcatnlo-pythia8.json",
                  ],
        "filter" : {
            "samples": ["TTTJ_TuneCP5_13TeV-madgraph-pythia8",
                        "TTTT_TuneCP5_13TeV-amcatnlo-pythia8",
                        "TWZ-Zto2Q_TuneCP5_13TeV_madgraph-pythia8",
                        "TWZToLL_thad_Wlept_5f_DR_TuneCP5_13TeV-amcatnlo-pythia8"
                        ],
            "samples_exclude" : [],
            "year": ["2016_PreVFP",
                     "2016_PostVFP",
                     "2017",
                     "2018"
                     ] #All the years
        }
    },

    workflow = ttHbbBaseProcessor,
    workflow_options = {},
    
    skim = [
            eventFlags,
            goldenJson,
            get_nPVgood(1),
            get_nObj_min(4, 15., "Jet"),
            get_nBtagMin(3, 15., coll="Jet", wp="M"),
            get_HLTsel(primaryDatasets=["SingleEle", "SingleMuon"])],
    
    preselections = [semileptonic_presel],
    categories = {
        "baseline": [passthrough],
        "4j3b": [get_nObj_min(4, minpt=30., coll="JetGood"), get_nBtagMin(3, minpt=30., coll="BJetGood")],
        "5j4b": [get_nObj_min(5, minpt=30., coll="JetGood"), get_nBtagMin(4, minpt=30., coll="BJetGood")],
        
    },

    weights= {
        "common": {
            "inclusive": [
                "genWeight", "lumi","XS",
                "pileup",
                "sf_ele_reco", "sf_ele_id",
                "sf_mu_id", "sf_mu_iso",
                "sf_jet_puId",
            ],
            "bycategory": {},
        },
        "bysample": {},
    },
    variations = {
        "weights": {"common": {"inclusive": [], "bycategory": {}}, "bysample": {}},
    },
    
    variables = {
        **count_hist("JetGood"),
        **count_hist("BJetGood"),
        **count_hist("LeptonGood"),
        **jet_hists("JetGood"),
        **jet_hists("BJetGood"),
        
    },
    columns = {},
)

# Registering custom functions
import cloudpickle
cloudpickle.register_pickle_by_value(custom_cut_functions)
cloudpickle.register_pickle_by_value(custom_cuts)
