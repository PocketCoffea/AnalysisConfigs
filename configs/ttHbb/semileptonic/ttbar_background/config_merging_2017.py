from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.cut_functions import get_nObj_eq, get_nObj_min, get_nObj_less, get_HLTsel, get_nBtagMin, get_nElectron, get_nMuon
from pocket_coffea.parameters.histograms import *
from pocket_coffea.parameters.cuts import passthrough

import workflow
from workflow import ttbarBackgroundProcessor

import custom_cut_functions
import custom_cuts
from custom_cut_functions import *
from custom_cuts import *
from params.binning import bins
from params.axis_settings import axis_settings
import os
localdir = os.path.dirname(os.path.abspath(__file__))

# Loading default parameters
from pocket_coffea.parameters import defaults
default_parameters = defaults.get_default_parameters()
defaults.register_configuration_dir("config_dir", localdir+"/params")

# Samples to exclude in specific histograms
exclude_data = ["DATA_SingleEle", "DATA_SingleMuon"]
exclude_nonttbar = ["ttHTobb", "TTTo2L2Nu", "SingleTop", "WJetsToLNu_HT"] + exclude_data

# Electron and muon categories
ele_categories_dict = {
    "SingleEle_>=4j==3b" : [ get_nElectron(1, coll="ElectronGood"), get_nObj_eq(3, coll="BJetGood") ],
    "SingleEle_>=4j>=4b" : [ get_nElectron(1, coll="ElectronGood"), get_nBtagMin(4, coll="BJetGood") ],
    "SingleEle_==4j>=4b" : [ get_nElectron(1, coll="ElectronGood"), get_nObj_eq(4, coll="JetGood"), get_nBtagMin(4, coll="BJetGood") ],
    "SingleEle_==5j>=4b" : [ get_nElectron(1, coll="ElectronGood"), get_nObj_eq(5, coll="JetGood"), get_nBtagMin(4, coll="BJetGood") ],
    "SingleEle_>=6j>=4b" : [ get_nElectron(1, coll="ElectronGood"), get_nObj_min(6, coll="JetGood"), get_nBtagMin(4, coll="BJetGood") ],
}

muon_categories_dict = {
    "SingleMuon_>=4j==3b" : [ get_nMuon(1, coll="MuonGood"), get_nObj_eq(3, coll="BJetGood") ],
    "SingleMuon_>=4j>=4b" : [ get_nMuon(1, coll="MuonGood"), get_nBtagMin(4, coll="BJetGood") ],
    "SingleMuon_==4j>=4b" : [ get_nMuon(1, coll="MuonGood"), get_nObj_eq(4, coll="JetGood"), get_nBtagMin(4, coll="BJetGood") ],
    "SingleMuon_==5j>=4b" : [ get_nMuon(1, coll="MuonGood"), get_nObj_eq(5, coll="JetGood"), get_nBtagMin(4, coll="BJetGood") ],
    "SingleMuon_>=6j>=4b" : [ get_nMuon(1, coll="MuonGood"), get_nObj_min(6, coll="JetGood"), get_nBtagMin(4, coll="BJetGood") ],
}

ele_categories = list(ele_categories_dict.keys())
muon_categories = list(muon_categories_dict.keys())

# adding object preselection
year = "2017"
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
                        "DATA_SingleMuon"
                        ],
            "samples_exclude" : [],
            "year": [year]
        },
        "subsamples": {
            'DATA_SingleEle'  : {
                'DATA_SingleEle' : [get_HLTsel(primaryDatasets=["SingleEle"])]
            },
            'DATA_SingleMuon' : {
                'DATA_SingleMuon' : [get_HLTsel(primaryDatasets=["SingleMuon"]),
                                     get_HLTsel(primaryDatasets=["SingleEle"], invert=True)]
            },
            'TTbbSemiLeptonic' : {
                'TTbbSemiLeptonic_tt+LF'   : [get_genTtbarId_100_eq(0)],
                #'TTbbSemiLeptonic_tt+c'    : [get_genTtbarId_100_eq(41)],
                #'TTbbSemiLeptonic_tt+2c'   : [get_genTtbarId_100_eq(42)],
                #'TTbbSemiLeptonic_tt+cc'   : [get_genTtbarId_100_eq(43)],
                #'TTbbSemiLeptonic_tt+c2c'  : [get_genTtbarId_100_eq(44)],
                #'TTbbSemiLeptonic_tt+2c2c' : [get_genTtbarId_100_eq(45)],
                #'TTbbSemiLeptonic_tt+C'    : [get_genTtbarId_100_eq(46)],
                'TTbbSemiLeptonic_tt+C'    : [get_genTtbarId_100_eq([41, 42, 43, 44, 45, 46])],
                #'TTbbSemiLeptonic_tt+b'    : [get_genTtbarId_100_eq(51)],
                #'TTbbSemiLeptonic_tt+2b'   : [get_genTtbarId_100_eq(52)],
                #'TTbbSemiLeptonic_tt+bb'   : [get_genTtbarId_100_eq(53)],
                #'TTbbSemiLeptonic_tt+b2b'  : [get_genTtbarId_100_eq(54)],
                #'TTbbSemiLeptonic_tt+2b2b' : [get_genTtbarId_100_eq(55)],
                #'TTbbSemiLeptonic_tt+B'    : [get_genTtbarId_100_eq(56)],
                'TTbbSemiLeptonic_tt+B'    : [get_genTtbarId_100_eq([51, 52, 53, 54, 55, 56])],
            },
            'TTToSemiLeptonic' : {
                'TTToSemiLeptonic_tt+LF'   : [get_genTtbarId_100_eq(0)],
                #'TTToSemiLeptonic_tt+c'    : [get_genTtbarId_100_eq(41)],
                #'TTToSemiLeptonic_tt+2c'   : [get_genTtbarId_100_eq(42)],
                #'TTToSemiLeptonic_tt+cc'   : [get_genTtbarId_100_eq(43)],
                #'TTToSemiLeptonic_tt+c2c'  : [get_genTtbarId_100_eq(44)],
                #'TTToSemiLeptonic_tt+2c2c' : [get_genTtbarId_100_eq(45)],
                #'TTToSemiLeptonic_tt+C'    : [get_genTtbarId_100_eq(46)],
                'TTToSemiLeptonic_tt+C'    : [get_genTtbarId_100_eq([41, 42, 43, 44, 45, 46])],
                #'TTToSemiLeptonic_tt+b'    : [get_genTtbarId_100_eq(51)],
                #'TTToSemiLeptonic_tt+2b'   : [get_genTtbarId_100_eq(52)],
                #'TTToSemiLeptonic_tt+bb'   : [get_genTtbarId_100_eq(53)],
                #'TTToSemiLeptonic_tt+b2b'  : [get_genTtbarId_100_eq(54)],
                #'TTToSemiLeptonic_tt+2b2b' : [get_genTtbarId_100_eq(55)],
                #'TTToSemiLeptonic_tt+B'    : [get_genTtbarId_100_eq(56)],
                'TTToSemiLeptonic_tt+B'    : [get_genTtbarId_100_eq([51, 52, 53, 54, 55, 56])],
            },
        }
    },

    workflow = ttbarBackgroundProcessor,
    workflow_options = {"parton_jet_min_dR": 0.3},
    
    skim = [get_nObj_min(4, 15., "Jet"),
            get_HLTsel(primaryDatasets=["SingleEle", "SingleMuon"])],
    preselections = [semileptonic_presel_nobtag],
    categories = {
        "baseline": [passthrough],
        **muon_categories_dict,
        **ele_categories_dict,
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
                "inclusive": ["pileup",
                              "sf_ele_reco", "sf_ele_id",
                              "sf_mu_id", "sf_mu_iso", "sf_mu_trigger",
                              "sf_jet_puId"
                             ]
                             + [f"sf_btag_{b}" for b in parameters["systematic_variations"]["weight_variations"]["sf_btag"][year]]
                             + [f"sf_ele_trigger_{v}" for v in parameters["systematic_variations"]["weight_variations"]["sf_ele_trigger"][year]],
                "bycategory" : {
                }
            },
            "bysample": {
            }
        },
        "shape": {
            "common":{
                "inclusive": [ "JES_Total_AK4PFchs", "JER_AK4PFchs" ]
            }
        }
    },
    
    variables = {
        **ele_hists(coll="ElectronGood", pos=0, axis_settings=axis_settings, exclude_categories=muon_categories),
        **muon_hists(coll="MuonGood", pos=0, axis_settings=axis_settings, exclude_categories=ele_categories),
        "ElectronGood_pt_1_rebin" : HistConf(
            [
                Axis(coll="ElectronGood", field="pt", pos=0, type="variable",
                     bins=bins["ElectronGood_pt"][year],
                     label="Electron $p_{T}$ [GeV]",
                     lim=(0,500))
            ],
            exclude_categories=muon_categories
        ),
        "ElectronGood_etaSC_1_rebin" : HistConf(
            [
                Axis(coll="ElectronGood", field="etaSC", pos=0, type="variable",
                     bins=bins["ElectronGood_etaSC"][year],
                     label="Electron Supercluster $\eta$",
                     lim=(-2.5,2.5))
            ],
            exclude_categories=muon_categories
        ),
        **count_hist(name="nLeptons", coll="LeptonGood",bins=3, start=0, stop=3),
        **count_hist(name="nJets", coll="JetGood",bins=10, start=4, stop=14),
        **count_hist(name="nBJets", coll="BJetGood",bins=10, start=0, stop=10),
        **count_hist(name="nGenJets", coll="GenJetGood", bins=14, start=0, stop=14, exclude_samples=exclude_data),
        **count_hist(name="nBGenJets", coll="BGenJetGood", bins=10, start=0, stop=10, exclude_samples=exclude_data),
        **count_hist(name="nCGenJets", coll="CGenJetGood", bins=10, start=0, stop=10, exclude_samples=exclude_data),
        **count_hist(name="nLGenJets", coll="LGenJetGood", bins=10, start=0, stop=10, exclude_samples=exclude_data),
        **count_hist(name="nBGenJetsExtra", coll="BGenJetGoodExtra", bins=10, start=0, stop=10, exclude_samples=exclude_nonttbar),
        **count_hist(name="nCGenJetsExtra", coll="CGenJetGoodExtra", bins=10, start=0, stop=10, exclude_samples=exclude_nonttbar),
        **jet_hists(coll="JetGood", pos=0, axis_settings=axis_settings),
        **jet_hists(coll="JetGood", pos=1, axis_settings=axis_settings),
        **jet_hists(coll="JetGood", pos=2, axis_settings=axis_settings),
        **jet_hists(coll="JetGood", pos=3, axis_settings=axis_settings),
        **jet_hists(coll="JetGood", pos=4, axis_settings=axis_settings),
        **jet_hists(name="bjet",coll="BJetGood", pos=0, axis_settings=axis_settings),
        **jet_hists(name="bjet",coll="BJetGood", pos=1, axis_settings=axis_settings),
        **jet_hists(name="bjet",coll="BJetGood", pos=2, axis_settings=axis_settings),
        **jet_hists(name="bjet",coll="BJetGood", pos=3, axis_settings=axis_settings),
        **jet_hists(name="bjet",coll="BJetGood", pos=4, axis_settings=axis_settings),
        **jet_hists(coll="GenJetGood", pos=0, fields=["pt", "eta", "phi"], axis_settings=axis_settings, exclude_samples=exclude_data),
        **jet_hists(coll="GenJetGood", pos=1, fields=["pt", "eta", "phi"], axis_settings=axis_settings, exclude_samples=exclude_data),
        **jet_hists(coll="GenJetGood", pos=2, fields=["pt", "eta", "phi"], axis_settings=axis_settings, exclude_samples=exclude_data),
        **jet_hists(coll="GenJetGood", pos=3, fields=["pt", "eta", "phi"], axis_settings=axis_settings, exclude_samples=exclude_data),
        **jet_hists(coll="GenJetGood", pos=4, fields=["pt", "eta", "phi"], axis_settings=axis_settings, exclude_samples=exclude_data),
        **jet_hists(name="bgenjet",coll="BGenJetGood", pos=0, fields=["pt", "eta", "phi"], axis_settings=axis_settings, exclude_samples=exclude_data),
        **jet_hists(name="bgenjet",coll="BGenJetGood", pos=1, fields=["pt", "eta", "phi"], axis_settings=axis_settings, exclude_samples=exclude_data),
        **jet_hists(name="bgenjet",coll="BGenJetGood", pos=2, fields=["pt", "eta", "phi"], axis_settings=axis_settings, exclude_samples=exclude_data),
        **jet_hists(name="bgenjet",coll="BGenJetGood", pos=3, fields=["pt", "eta", "phi"], axis_settings=axis_settings, exclude_samples=exclude_data),
        **jet_hists(name="bgenjet",coll="BGenJetGood", pos=4, fields=["pt", "eta", "phi"], axis_settings=axis_settings, exclude_samples=exclude_data),
        **jet_hists(name="cgenjet",coll="CGenJetGood", pos=0, fields=["pt", "eta", "phi"], axis_settings=axis_settings, exclude_samples=exclude_data),
        **jet_hists(name="cgenjet",coll="CGenJetGood", pos=1, fields=["pt", "eta", "phi"], axis_settings=axis_settings, exclude_samples=exclude_data),
        **jet_hists(name="cgenjet",coll="CGenJetGood", pos=2, fields=["pt", "eta", "phi"], axis_settings=axis_settings, exclude_samples=exclude_data),
        **jet_hists(name="cgenjet",coll="CGenJetGood", pos=3, fields=["pt", "eta", "phi"], axis_settings=axis_settings, exclude_samples=exclude_data),
        **jet_hists(name="cgenjet",coll="CGenJetGood", pos=4, fields=["pt", "eta", "phi"], axis_settings=axis_settings, exclude_samples=exclude_data),
        **jet_hists(name="bgenjet_extra",coll="BGenJetGoodExtra", pos=0, fields=["pt", "eta", "phi"], axis_settings=axis_settings, exclude_samples=exclude_nonttbar),
        **jet_hists(name="bgenjet_extra",coll="BGenJetGoodExtra", pos=1, fields=["pt", "eta", "phi"], axis_settings=axis_settings, exclude_samples=exclude_nonttbar),
        **jet_hists(name="bgenjet_extra",coll="BGenJetGoodExtra", pos=2, fields=["pt", "eta", "phi"], axis_settings=axis_settings, exclude_samples=exclude_nonttbar),
        **jet_hists(name="bgenjet_extra",coll="BGenJetGoodExtra", pos=3, fields=["pt", "eta", "phi"], axis_settings=axis_settings, exclude_samples=exclude_nonttbar),
        **jet_hists(name="bgenjet_extra",coll="BGenJetGoodExtra", pos=4, fields=["pt", "eta", "phi"], axis_settings=axis_settings, exclude_samples=exclude_nonttbar),
        **jet_hists(name="cgenjet_extra",coll="CGenJetGoodExtra", pos=0, fields=["pt", "eta", "phi"], axis_settings=axis_settings, exclude_samples=exclude_nonttbar),
        **jet_hists(name="cgenjet_extra",coll="CGenJetGoodExtra", pos=1, fields=["pt", "eta", "phi"], axis_settings=axis_settings, exclude_samples=exclude_nonttbar),
        **jet_hists(name="cgenjet_extra",coll="CGenJetGoodExtra", pos=2, fields=["pt", "eta", "phi"], axis_settings=axis_settings, exclude_samples=exclude_nonttbar),
        **jet_hists(name="cgenjet_extra",coll="CGenJetGoodExtra", pos=3, fields=["pt", "eta", "phi"], axis_settings=axis_settings, exclude_samples=exclude_nonttbar),
        **jet_hists(name="cgenjet_extra",coll="CGenJetGoodExtra", pos=4, fields=["pt", "eta", "phi"], axis_settings=axis_settings, exclude_samples=exclude_nonttbar),
        **met_hists(coll="MET", axis_settings=axis_settings),
        "deltaRbb_min" : HistConf(
            [Axis(coll="events", field="deltaRbb_min", bins=50, start=0, stop=5,
                  label="$\Delta R_{bb}$")]
        ),
        "deltaEtabb_min" : HistConf(
            [Axis(coll="events", field="deltaEtabb_min", bins=50, start=0, stop=5,
                  label="$\Delta \eta_{bb}$")]
        ),
        "deltaPhibb_min" : HistConf(
            [Axis(coll="events", field="deltaPhibb_min", bins=50, start=0, stop=5,
                  label="$\Delta \phi_{bb}$")]
        ),
        "mbb" : HistConf(
            [Axis(coll="events", field="mbb", bins=50, start=0, stop=500,
                    label="$m_{bb}$ [GeV]")]
        ),
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
            exclude_categories=muon_categories
        ),
    },
)

run_options = {
        "executor"       : "dask/slurm",
        "env"            : "conda",
        "workers"        : 1,
        "scaleout"       : 200,
        "worker_image"   : "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/pocketcoffea:lxplus-cc7-latest",
        "queue"          : "standard",
        "walltime"       : "12:00:00",
        "mem_per_worker" : "6GB", # GB
        "disk_per_worker" : "1GB", # GB
        "exclusive"      : False,
        "chunk"          : 400000,
        "retries"        : 50,
        "treereduction"  : 5,
        "adapt"          : False,
    }


if "dask"  in run_options["executor"]:
    import cloudpickle
    cloudpickle.register_pickle_by_value(workflow)
    cloudpickle.register_pickle_by_value(custom_cut_functions)
    cloudpickle.register_pickle_by_value(custom_cuts)
