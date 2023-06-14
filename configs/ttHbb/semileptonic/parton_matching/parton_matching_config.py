from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.cut_definition import Cut
from pocket_coffea.lib.columns_manager import ColOut
from pocket_coffea.lib.cut_functions import get_nObj_min, get_HLTsel, get_nBtagMin
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.parameters.histograms import *

import workflow
from workflow import PartonMatchingProcessor

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
        "jsons": [f"{localdir}/datasets/signal_ttHTobb_local.json"],
        "filter" : {
            "samples": ["ttHTobb"],
            "samples_exclude" : [],
            "year": ["2016_PreVFP",
                     "2016_PostVFP",
                     "2017","2018"] #All the years
        }
    },

    workflow = PartonMatchingProcessor,
    workflow_options = {"parton_jet_min_dR": 0.3},
    
    skim = [get_nObj_min(4, 15., coll="Jet"),
            get_nBtagMin(3, 15., coll="Jet"),
            get_HLTsel()], 
    
    preselections = [semileptonic_presel],
    categories = {
        "baseline": [passthrough],
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
    
    variables = {
        **jet_hists(coll="JetGood"),
        **jet_hists(coll="BJetGood"),
        **ele_hists(coll="ElectronGood"),
        **muon_hists(coll="MuonGood"),
        **count_hist(name="nJets", coll="JetGood", bins=10, start=4, stop=14),
        **count_hist(name="nBJets", coll="BJetGood", bins=12, start=2, stop=14),
        **jet_hists(coll="JetGood", pos=0),
        **jet_hists(coll="JetGood", pos=1),
        **jet_hists(coll="JetGood", pos=2),
        **jet_hists(coll="JetGood", pos=3),
        **jet_hists(coll="JetGood", pos=4),
        **jet_hists(name="bjet", coll="BJetGood", pos=0),
        **jet_hists(name="bjet", coll="BJetGood", pos=1),
        **jet_hists(name="bjet", coll="BJetGood", pos=2),
        **jet_hists(name="bjet", coll="BJetGood", pos=3),
        **jet_hists(name="bjet", coll="BJetGood", pos=4),
        **count_hist(name="nPartons", coll="Parton", bins=10, start=0, stop=10),
        **count_hist(
            name="nPartonsMatched", coll="PartonMatched", bins=10, start=0, stop=10
        ),
        **parton_hists(coll="Parton", fields=["pt", "eta", "phi", "pdgId"]),
        **parton_hists(coll="PartonMatched"),
        "NJet_Nparton_total": HistConf(
            [
                Axis(
                    coll="events",
                    field="nJetGood",
                    bins=11,
                    start=4,
                    stop=15,
                    label="N. jets",
                ),
                Axis(
                    coll="events",
                    field="nParton",
                    bins=11,
                    start=4,
                    stop=15,
                    label="N. partons",
                ),
            ],
            variations=False,
            no_weights=True,
        ),
        "NJet_Nparton_matched": HistConf(
            [
                Axis(
                    coll="events",
                    field="nJetGood",
                    bins=11,
                    start=4,
                    stop=15,
                    label="N. jets",
                ),
                Axis(
                    coll="events",
                    field="nPartonMatched",
                    bins=15,
                    start=0,
                    stop=15,
                    label="N. partons matched",
                ),
            ],
            variations=False,
            no_weights=True,
        ),
        "Nparton_Nparton_matched": HistConf(
            [
                Axis(
                    coll="events",
                    field="nParton",
                    bins=11,
                    start=4,
                    stop=15,
                    label="N. partons",
                ),
                Axis(
                    coll="events",
                    field="nPartonMatched",
                    bins=15,
                    start=0,
                    stop=15,
                    label="N. partons matched",
                ),
            ],
            variations=False,
            no_weights=True,
        ),
        "hist_ptComparison_parton_matching": HistConf(
            [
                Axis(
                    coll="JetGoodMatched",
                    field="pt",
                    bins=40,
                    start=0,
                    stop=300,
                    label="Jet Pt",
                ),
                Axis(
                    coll="PartonMatched",
                    field="pt",
                    bins=40,
                    start=0,
                    stop=300,
                    label="Parton Pt",
                ),
                Axis(
                    coll="JetGoodMatched",
                    field="eta",
                    bins=5,
                    start=-2.4,
                    stop=2.4,
                    label="Jet $\eta$",
                ),
            ],
            variations=False,
            no_weights=True,
        ),
    },
    columns = {
        "common": {
            "inclusive": [],
        },
        "bysample": {
            "ttHTobb": {
                "bycategory": {
                    "semilep_LHE": [
                        ColOut("Parton", ["pt", "eta", "phi", "pdgId", "provenance"]),
                        ColOut(
                            "PartonMatched",
                            ["pt", "eta", "phi", "pdgId", "provenance", "dRMatchedJet"],
                        ),
                        ColOut(
                            "JetGood",
                            ["pt", "eta", "phi", "hadronFlavour", "btagDeepFlavB"],
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
                            ],
                        ),
                        ColOut("LeptonGood",
                               ["pt","eta","phi"],
                               pos_end=1)
                    ]
                }
            }
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
        "mem_per_worker" : "8GB", # GB
        "disk_per_worker" : "1GB", # GB
        "exclusive"      : False,
        "chunk"          : 400000,
        "retries"        : 50,
        "treereduction"  : 20,
        "adapt"          : False,
        
    }
   


if "dask"  in run_options["executor"]:
    import cloudpickle
    cloudpickle.register_pickle_by_value(workflow)
    cloudpickle.register_pickle_by_value(custom_cut_functions)
    cloudpickle.register_pickle_by_value(custom_cuts)
