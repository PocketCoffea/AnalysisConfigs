from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.cut_definition import Cut
from pocket_coffea.lib.columns_manager import ColOut
from pocket_coffea.lib.cut_functions import get_nObj_min, get_HLTsel, get_nBtagMin
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.parameters.histograms import *

import workflow_improved
from workflow_improved import PartonMatchingProcessorWithFSR

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
                                                  f"{localdir}/params/object_preselection_semileptonic.yaml",
                                                  f"{localdir}/params/triggers.yaml",
                                                  f"{localdir}/params/lepton_scale_factors.yaml",
                                                  f"{localdir}/params/plotting_style.yaml",
                                                  update=True)

cfg = Configurator(
    parameters = parameters,
    datasets = {
        "jsons": [f"{localdir}/datasets/signal_ttHTobb_local.json",
                  f"{localdir}/datasets/signal_ttHTobb_ttToSemiLep_local.json"
                  ],
        "filter" : {
            "samples": ["ttHTobb",
                        "ttHTobb_ttToSemiLep"
                        ],
            "samples_exclude" : [],
            "year": ["2016_PreVFP",
                     "2016_PostVFP",
                     "2017",
                     "2018"
                     ] #All the years
        }
    },

    workflow = PartonMatchingProcessorWithFSR,
    workflow_options = {"parton_jet_min_dR": 0.3,
                        "parton_jet_min_dR_postfsr": 1.0},
    
    skim = [get_nObj_min(4, 15., "Jet"),
            get_nBtagMin(3, 15., coll="Jet", wp="M"),
            get_HLTsel(primaryDatasets=["SingleEle", "SingleMuon"])],
    
    preselections = [semileptonic_presel],
    categories = {
        "semilep_LHE": [semilep_lhe]
    },

    weights= {
        "common": {
            "inclusive": [
                "genWeight", "lumi","XS",
                "pileup",
                "sf_ele_reco", "sf_ele_id",
                "sf_mu_id", "sf_mu_iso",
                "sf_btag",
                "sf_jet_puId",
            ],
            "bycategory": {},
        },
        "bysample": {},
    },
    variations = {
        "weights": {"common": {"inclusive": [], "bycategory": {}}, "bysample": {}},
    },
    
    variables = {},
    columns = {
        "common": {
            "inclusive": [],
            "bycategory": {
                    "semilep_LHE": [
                        ColOut("PartonInitial", ["pt", "eta", "phi", "mass", "pdgId", "provenance"], flatten=False),
                        ColOut("PartonLastCopy",["pt", "eta", "phi","mass", "pdgId", "provenance",], flatten=False),
                        ColOut("PartonLastCopyMatched",["pt", "eta", "phi","mass", "pdgId", "provenance",], flatten=False),
                        ColOut(
                            "JetGood",
                            ["pt", "eta", "phi", "hadronFlavour", "btagDeepFlavB", "btag_L", "btag_M", "btag_H"], flatten=False
                        ),
                        ColOut(
                            "JetGoodMatched",
                            ["pt", "eta", "phi", "hadronFlavour", "btagDeepFlavB", "btag_L", "btag_M", "btag_H", "dRMatchedJet", "provenance"], flatten=False
                        ),
                        ColOut("LeptonGood",
                               ["pt","eta","phi", "pdgId", "charge", "mvaTTH"],
                               pos_end=1, store_size=False),
                        ColOut("MET", ["phi","pt","significance"]),
                        ColOut("Generator",["x1","x2","id1","id2","xpdf1","xpdf2"]),
                        ColOut("LeptonParton",["pt","eta","phi","mass","pdgId"]),
                    ]
            }
        },
        "bysample": {
            "ttHTobb": {
                "bycategory": {
                    "semilep_LHE": [
                        ColOut("HiggsGen",
                                ["pt","eta","phi","mass","pdgId"],
                                pos_end=1, store_size=False, flatten=False),
                        ColOut("TopGen",
                                ["pt","eta","phi","mass","pdgId"],
                                pos_end=1, store_size=False, flatten=False),
                        ColOut("AntiTopGen",
                                ["pt","eta","phi","mass","pdgId"],
                                pos_end=1, store_size=False, flatten=False)
                    ]
                }
            },
            "ttHTobb_ttToSemiLep": {
                "bycategory": {
                    "semilep_LHE": [
                        ColOut("HiggsGen",
                                ["pt","eta","phi","mass","pdgId"],
                                pos_end=1, store_size=False, flatten=False),
                        ColOut("TopGen",
                                ["pt","eta","phi","mass","pdgId"],
                                pos_end=1, store_size=False, flatten=False),
                        ColOut("AntiTopGen",
                                ["pt","eta","phi","mass","pdgId"],
                                pos_end=1, store_size=False, flatten=False)
                    ]
                }
            }
        },
    },
)

# Registering custom functions
import cloudpickle
cloudpickle.register_pickle_by_value(workflow_improved)
cloudpickle.register_pickle_by_value(custom_cut_functions)
cloudpickle.register_pickle_by_value(custom_cuts)
