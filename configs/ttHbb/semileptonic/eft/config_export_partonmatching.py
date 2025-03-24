from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.cut_definition import Cut
from pocket_coffea.lib.columns_manager import ColOut
from pocket_coffea.lib.cut_functions import  get_nObj_eq, get_nObj_min, get_HLTsel, get_nBtagMin, get_nPVgood, goldenJson, eventFlags
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.parameters.histograms import *
from pocket_coffea.lib.weights.common.common import common_weights
from pocket_coffea.lib.weights.common.weights_run2_UL import SF_ele_trigger, SF_QCD_renorm_scale, SF_QCD_factor_scale

import configs
import eft_weights
import workflow_EFT
import custom_cuts
from custom_cuts import semilep_lhe, semilep_lhe_notau
from workflow_EFT import ttHbbEFTProcessor
import configs.ttHbb.semileptonic.common.cuts.custom_cut_functions as custom_cut_functions
import configs.ttHbb.semileptonic.common.cuts.custom_cuts as custom_cuts
from configs.ttHbb.semileptonic.common.cuts.custom_cut_functions import *
from configs.ttHbb.semileptonic.common.cuts.custom_cuts import *
from configs.ttHbb.semileptonic.common.weights.custom_weights import SF_top_pt, SF_LHE_pdf_weight

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
                                                  f"{localdir}/params/btagging.yaml",
                                                  f"{localdir}/params/btagSF_calibration.yaml",
                                                  f"{localdir}/params/eft_params.yaml",
                                                  update=True)

# Weight for SM
SMEFT_SM_weight = eft_weights.getSMEFTweight([0.,0.,0.,0.,0.,0.,0.,0.], "SMEFT_toSM")


cfg = Configurator(
    parameters = parameters,
    datasets = {
        "jsons": [
            f"{localdir}/datasets/signal_ttHTobb_forTraining.json",
            f"{localdir}/datasets/datasets_skim_Run2.json",
            f"{localdir}/datasets/signal_ttHTobb_EFT.json",
        
        ],
        "filter" : {
            "samples": ["ttHTobb", #"TTToSemiLeptonic", "TTbbSemiLeptonic"
                        "ttHTobb_EFT"],
            "samples_exclude" : [],
            "year": [#"2016_PreVFP",
                     #"2016_PostVFP",
                     #"2017",
                     "2018"] #All the years
        }
    },

    workflow = ttHbbEFTProcessor,
    workflow_options = {"parton_jet_max_dR": 0.3,
                        "parton_jet_max_dR_postfsr": 1.0,
                        "dump_columns_as_arrays_per_chunk": "root://eoscms.cern.ch//eos/cms/store/group/phys_higgs/ttHbb/ntuples/EFT_export/"},
    
    skim = [
        get_nPVgood(1),
        eventFlags,
        goldenJson,
        get_nObj_min(4, 15., coll="Jet"),
        get_nBtagMin(3, 15., coll="Jet", wp="M"),
        get_HLTsel(["SingleEle", "SingleMuon"])], 
    
    preselections = [semileptonic_presel],
    categories = {
        "baseline": [semilep_lhe],
    },

    weights= {
        "common": {
            "inclusive": [
                "genWeight", "lumi","XS",
                "pileup",
                "sf_ele_reco", "sf_ele_id",
                "sf_mu_id", "sf_mu_iso", "sf_mu_trigger",
                "sf_btag",
                #"sf_btag_calib", TO BE UPDATED
                "sf_jet_puId","sf_top_pt",  "sf_partonshower_isr", "sf_partonshower_fsr"
            ],
            "bycategory": {},
        },
        "bysample": {
            "ttHTobb_EFT": {
                "inclusive" : ["SMEFT_toSM"]
            },
        }
    },
    weights_classes = common_weights + [SF_ele_trigger, SF_top_pt,SMEFT_SM_weight],
    variations = {
        "weights": {"common": {"inclusive": [], "bycategory": {}}, "bysample": {}},
    },
    
    variables = {},
    columns = {
        "common": {
            "bycategory": {
                    "baseline": [
                        ColOut("PartonInitial", ["pt", "eta", "phi", "mass", "pdgId", "provenance"], flatten=False),
                        ColOut("PartonLastCopy",["pt", "eta", "phi","mass", "pdgId", "provenance",], flatten=False),
                        ColOut("PartonInitialMatched",["pt", "eta", "phi","mass", "pdgId", "provenance",], flatten=False),
                        ColOut("PartonLastCopyMatched",["pt", "eta", "phi","mass", "pdgId", "provenance",], flatten=False),
                        ColOut(
                            "JetGood",
                            ["pt", "eta", "phi", "hadronFlavour", "btagDeepFlavB"], flatten=False
                        ),
                        ColOut(
                            "JetGoodMatched",
                            [
                                "pt",
                                "eta",
                                "phi",
                                "hadronFlavour",
                                "btagDeepFlavB",
                                "dRMatchedJet",
                                "provenance"
                            ], flatten=False
                        ),
                        
                        ColOut("LeptonGood",
                               ["pt","eta","phi"],flatten=False,
                               pos_end=1, store_size=False),
                        ColOut("MET", ["phi","pt","significance"], flatten=False),
                        ColOut("Generator",["x1","x2","id1","id2","xpdf1","xpdf2"], flatten=False),
                        ColOut("LeptonGenLevel",["pt","eta","phi","mass","pdgId"], flatten=False),
                        ColOut("HiggsGen",
                               ["pt","eta","phi","mass","pdgId"],
                               pos_end=1, store_size=False, flatten=False),
                        ColOut("TopGen",
                               ["pt","eta","phi","mass","pdgId"],
                               pos_end=1, store_size=False, flatten=False),
                        ColOut("AntiTopGen",
                               ["pt","eta","phi","mass","pdgId"],
                               pos_end=1, store_size=False, flatten=False),
                        ColOut("ISR",
                               ["pt","eta","phi","mass","pdgId"],
                                 pos_end=1, store_size=False, flatten=False),
                    ]
                }
        },
        "bysample":{}
    },
)

import cloudpickle
cloudpickle.register_pickle_by_value(workflow_EFT)
cloudpickle.register_pickle_by_value(custom_cuts)
cloudpickle.register_pickle_by_value(configs)
cloudpickle.register_pickle_by_value(eft_weights)
