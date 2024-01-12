from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.cut_functions import get_nObj_min, get_nObj_eq, get_HLTsel, get_nBtagMin, get_nElectron, get_nMuon
from pocket_coffea.parameters.histograms import *
from pocket_coffea.parameters.cuts import passthrough

import workflow
from workflow import ttbarBackgroundProcessor

import custom_cut_functions
import custom_cuts
from custom_cut_functions import *
from custom_cuts import *
from params.binning import bins
import os
localdir = os.path.dirname(os.path.abspath(__file__))

# Loading default parameters
from pocket_coffea.parameters import defaults
default_parameters = defaults.get_default_parameters()
defaults.register_configuration_dir("config_dir", localdir+"/params")

# adding object preselection
year = "2018"
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
                  f"{localdir}/datasets/backgrounds_MC_TTbb_local.json",
                  f"{localdir}/datasets/backgrounds_MC_ttbar_local.json",
                  f"{localdir}/datasets/backgrounds_MC_local.json",
                  f"{localdir}/datasets/DATA_SingleEle_local.json",
                  f"{localdir}/datasets/DATA_SingleMuon_local.json",
                    ],
        "filter" : {
            "samples": ["ttHTobb",
                        "TTbbSemiLeptonic",
                        "TTToSemiLeptonic",
                        "TTTo2L2Nu",
                        "SingleTop",
                        "WJetsToLNu_HT",
                        "DATA_SingleEle",
                        "DATA_SingleMuon"],
            "samples_exclude" : [],
            "year": [year]
        },
        "subsamples": {
            'DATA_SingleEle'  : {'DATA_SingleEle' : [get_HLTsel(primaryDatasets=["SingleEle"])]},
            'DATA_SingleMuon' : {'DATA_SingleMuon' : [get_HLTsel(primaryDatasets=["SingleMuon"]),
                                                      get_HLTsel(primaryDatasets=["SingleEle"], invert=True)]}
        }
    },

    workflow = ttbarBackgroundProcessor,
    
    skim = [get_nObj_min(4, 15., "Jet"),
            get_HLTsel(primaryDatasets=["SingleEle", "SingleMuon"])],
    preselections = [semileptonic_presel_nobtag,
                     get_nObj_eq(1, coll="BJetGood")],
    categories = {
        "baseline": [passthrough],
        "SingleEle_1b" : [ get_nElectron(1, coll="ElectronGood"), get_nBtagMin(1, coll="BJetGood") ],
        "SingleMuon_1b" : [ get_nMuon(1, coll="MuonGood"), get_nBtagMin(1, coll="BJetGood") ],
    },

    weights = {
        "common": {
            "inclusive": ["genWeight","lumi","XS",
                          "pileup",
                          "sf_ele_reco", "sf_ele_id", "sf_ele_trigger",
                          "sf_mu_id","sf_mu_iso", "sf_mu_trigger",
                          "sf_btag", #"sf_btag_calib",
                          "sf_jet_puId"
                          ],
            "bycategory" : {
            }
        },
        "bysample": {
        }
    },

    variations = {
        "weights": {
            "common": {
                "inclusive": [  "pileup",
                                "sf_ele_reco", "sf_ele_id",
                                "sf_mu_id", "sf_mu_iso", "sf_mu_trigger",
                                "sf_jet_puId"
                              ],# + [ f"sf_btag_{b}" for b in parameters["systematic_variations"]["weight_variations"]["sf_btag"][year]],
                              #+ [f"sf_ele_trigger_{v}" for v in parameters["systematic_variations"]["weight_variations"]["sf_ele_trigger"][year]],
                "bycategory" : {
                }
            },
            "bysample": {
            }
        },
        "shape": {
            "common":{
                "inclusive": [ ]
            }
        }
    },
    
    variables = {
        **ele_hists(coll="ElectronGood", pos=0, exclude_categories=["SingleMuon_1b", "SingleMuon_2b", "SingleMuon_3b", "SingleMuon_4b"]),
        **muon_hists(coll="MuonGood", pos=0, exclude_categories=["SingleEle_1b", "SingleEle_2b", "SingleEle_3b", "SingleEle_4b"]),
        "ElectronGood_pt_1_rebin" : HistConf(
            [
                Axis(coll="ElectronGood", field="pt", pos=0, type="variable",
                     bins=bins["ElectronGood_pt"][year],
                     label="Electron $p_{T}$ [GeV]",
                     lim=(0,500))
            ],
            exclude_categories=["SingleMuon_1b", "SingleMuon_2b", "SingleMuon_3b", "SingleMuon_4b"]
        ),
        "ElectronGood_etaSC_1_rebin" : HistConf(
            [
                Axis(coll="ElectronGood", field="etaSC", pos=0, type="variable",
                     bins=bins["ElectronGood_etaSC"][year],
                     label="Electron Supercluster $\eta$",
                     lim=(-2.5,2.5))
            ],
            exclude_categories=["SingleMuon_1b", "SingleMuon_2b", "SingleMuon_3b", "SingleMuon_4b"]
        ),
        **count_hist(name="nLeptons", coll="LeptonGood",bins=3, start=0, stop=3),
        **count_hist(name="nJets", coll="JetGood",bins=10, start=4, stop=14),
        **count_hist(name="nBJets", coll="BJetGood",bins=14, start=0, stop=14),
        **jet_hists(coll="JetGood", pos=0),
        **jet_hists(coll="JetGood", pos=1),
        **jet_hists(coll="JetGood", pos=2),
        **jet_hists(coll="JetGood", pos=3),
        **jet_hists(coll="JetGood", pos=4),
        **jet_hists(name="bjet",coll="BJetGood", pos=0),
        **met_hists(coll="MET"),
        "jets_Ht" : HistConf(
          [Axis(coll="events", field="JetGood_Ht", bins=100, start=0, stop=2500,
                label="Jets $H_T$ [GeV]")]
        ),
        "electron_etaSC_pt_leading" : HistConf(
            [
                Axis(coll="ElectronGood", field="pt", pos=0, type="variable",
                     bins=bins["ElectronGood_pt"][year],
                     label="Electron $p_{T}$ [GeV]",
                     lim=(0,500)),
                Axis(coll="ElectronGood", field="etaSC", pos=0, type="variable",
                     bins=bins["ElectronGood_etaSC"][year],
                     label="Electron Supercluster $\eta$",
                     lim=(-2.5,2.5)),
            ],
            exclude_categories=["SingleMuon_1b", "SingleMuon_2b", "SingleMuon_3b", "SingleMuon_4b"]
        ),
    },
)

run_options = {
        "executor"       : "dask/slurm",
        "env"            : "conda",
        "workers"        : 1,
        "scaleout"       : 150,
        "worker_image"   : "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/pocketcoffea:lxplus-cc7-latest",
        "queue"          : "standard",
        "walltime"       : "12:00:00",
        "mem_per_worker" : "6GB", # GB
        "disk_per_worker" : "1GB", # GB
        "exclusive"      : False,
        "chunk"          : 200000,
        "retries"        : 50,
        "treereduction"  : 20,
        "adapt"          : False,
        
    }


if "dask"  in run_options["executor"]:
    import cloudpickle
    cloudpickle.register_pickle_by_value(workflow)
    cloudpickle.register_pickle_by_value(custom_cut_functions)
    cloudpickle.register_pickle_by_value(custom_cuts)
