from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.cut_functions import get_nObj_min, get_HLTsel
from pocket_coffea.parameters.histograms import *
from pocket_coffea.parameters.cuts import passthrough
from math import pi

import workflow
from workflow import semileptonicTriggerProcessor

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

# adding object preselection
parameters = defaults.merge_parameters_from_files(default_parameters,
                                                  f"{localdir}/params/object_preselection_semileptonic.yaml",
                                                  f"{localdir}/params/triggers.yaml",
                                                  update=True)


cfg = Configurator(
    parameters = parameters,
    datasets = {
        "jsons": [f"{localdir}/datasets/backgrounds_MC_ttbar.json",
                  f"{localdir}/datasets/DATA_SingleMuon.json",
                    ],
        "filter" : {
            "samples": ["TTToSemiLeptonic",
                        "TTTo2L2Nu",
                        "DATA_SingleMuon"],
            "samples_exclude" : [],
            "year": ['2018']
        },
    },

    workflow = semileptonicTriggerProcessor,
    
    skim = [get_nObj_min(3, 15., "Jet"),
            get_HLTsel(primaryDatasets=["SingleMuon"])],
    preselections = [semileptonic_presel_triggerSF],
    categories = {
        "Ele32_EleHT_pass" : [
            get_HLTsel(primaryDatasets=["SingleEle"])
        ],
        "Ele32_EleHT_fail" : [
            get_HLTsel(primaryDatasets=["SingleEle"], invert=True)
        ],
        "Ele32_EleHT_pass_lowHT" : [
            get_HLTsel(primaryDatasets=["SingleEle"]),
            get_ht_below(400)
        ],
        "Ele32_EleHT_fail_lowHT" : [
            get_HLTsel(primaryDatasets=["SingleEle"], invert=True),
            get_ht_below(400)
        ],
        "Ele32_EleHT_pass_highHT" : [
            get_HLTsel(primaryDatasets=["SingleEle"]),
            get_ht_above(400)
        ],
        "Ele32_EleHT_fail_highHT" : [
            get_HLTsel(primaryDatasets=["SingleEle"], invert=True),
            get_ht_above(400)
        ],
        "inclusive" : [passthrough],
    },

    weights = {
        "common": {
            "inclusive": ["genWeight","lumi","XS",
                          "pileup",
                          "sf_ele_reco", "sf_ele_id",
                          "sf_mu_id","sf_mu_iso",
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
                "inclusive": [ "pileup" ],
                "bycategory" : {
                }
            },
        "bysample": {
        }    
        },
    },
    
   variables = {
        **muon_hists(coll="MuonGood"),
        **muon_hists(coll="MuonGood", pos=0),
        "ElectronGood_pt" : HistConf(
            [
                Axis(coll="ElectronGood", field="pt", type="variable",
                     bins=[30, 35, 40, 50, 60, 70, 80, 90, 100, 200, 500],
                     label="Electron $p_{T}$ [GeV]",
                     lim=(0,500))
            ]
        ),
        "ElectronGood_etaSC" : HistConf(
            [
                Axis(coll="ElectronGood", field="etaSC", type="variable",
                     bins=[-2.5, -2.0, -1.5660, -1.4442, -1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4442, 1.5660, 2.0, 2.5],
                     label="Electron Supercluster $\eta$",
                     lim=(-2.5,2.5))
            ]
        ),
        "ElectronGood_phi" : HistConf(
            [
                Axis(coll="ElectronGood", field="phi",
                     bins=12, start=-pi, stop=pi,
                     label="Electron $\phi$"),
            ]
        ),
        "ElectronGood_pt_1" : HistConf(
            [
                Axis(coll="ElectronGood", field="pt", pos=0, type="variable",
                     bins=[30, 35, 40, 50, 60, 70, 80, 90, 100, 200, 500],
                     label="Electron $p_{T}$ [GeV]",
                     lim=(0,500))
            ]
        ),
        "ElectronGood_etaSC_1" : HistConf(
            [
                Axis(coll="ElectronGood", field="etaSC", pos=0, type="variable",
                     bins=[-2.5, -2.0, -1.5660, -1.4442, -1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4442, 1.5660, 2.0, 2.5],
                     label="Electron Supercluster $\eta$",
                     lim=(-2.5,2.5))
            ]
        ),
        "ElectronGood_phi_1" : HistConf(
            [
                Axis(coll="ElectronGood", field="phi", pos=0,
                     bins=12, start=-pi, stop=pi,
                     label="Electron $\phi$"),
            ]
        ),
        **jet_hists(coll="JetGood"),
        **count_hist(name="nMuons", coll="MuonGood",bins=10, start=0, stop=2),
        **count_hist(name="nElectrons", coll="ElectronGood",bins=10, start=0, stop=2),
        **count_hist(name="nLeptons", coll="LeptonGood",bins=10, start=0, stop=2),
        **count_hist(name="nJets", coll="JetGood",bins=10, start=4, stop=10),
        **count_hist(name="nBJets", coll="BJetGood",bins=10, start=4, stop=10),
        "ht" : HistConf(
            [
                Axis(coll="events", field="JetGood_Ht", bins=400, start=0, stop=4000, label="$H_T$", lim=(0,1000))
            ]
        ),
        "electron_etaSC_pt_leading" : HistConf(
            [
                Axis(coll="ElectronGood", field="pt", pos=0, type="variable",
                     bins=[30, 35, 40, 50, 60, 70, 80, 90, 100, 200, 500],
                     label="Electron $p_{T}$ [GeV]",
                     lim=(0,500)),
                Axis(coll="ElectronGood", field="etaSC", pos=0, type="variable",
                     bins=[-2.5, -2.0, -1.5660, -1.4442, -1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4442, 1.5660, 2.0, 2.5],
                     label="Electron Supercluster $\eta$",
                     lim=(-2.5,2.5)),
            ]
        ),
        "electron_phi_pt_leading" : HistConf(
            [
                Axis(coll="ElectronGood", field="pt", pos=0, type="variable",
                     bins=[30, 35, 40, 50, 60, 70, 80, 90, 100, 200, 500],
                     label="Electron $p_{T}$ [GeV]",
                     lim=(0,500)),
                Axis(coll="ElectronGood", field="phi", pos=0,
                     bins=12, start=-pi, stop=pi,
                     label="Electron $\phi$"),
            ]
        ),
        "electron_etaSC_phi_leading" : HistConf(
            [
                Axis(coll="ElectronGood", field="phi", pos=0,
                     bins=12, start=-pi, stop=pi,
                     label="Electron $\phi$"),
                Axis(coll="ElectronGood", field="etaSC", pos=0, type="variable",
                     bins=[-2.5, -2.0, -1.5660, -1.4442, -1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4442, 1.5660, 2.0, 2.5],
                     label="Electron Supercluster $\eta$",
                     lim=(-2.5,2.5)),
            ]
        ),
        "electron_etaSC_pt_all" : HistConf(
            [
                Axis(coll="ElectronGood", field="pt", type="variable",
                     bins=[30, 35, 40, 50, 60, 70, 80, 90, 100, 200, 500],
                     label="Electron $p_{T}$ [GeV]",
                     lim=(0,500)),
                Axis(coll="ElectronGood", field="etaSC", type="variable",
                     bins=[-2.5, -2.0, -1.5660, -1.4442, -1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4442, 1.5660, 2.0, 2.5],
                     label="Electron Supercluster $\eta$",
                     lim=(-2.5,2.5)),
            ]
        ),
        "electron_phi_pt_all" : HistConf(
            [
                Axis(coll="ElectronGood", field="pt", type="variable",
                     bins=[30, 35, 40, 50, 60, 70, 80, 90, 100, 200, 500],
                     label="Electron $p_{T}$ [GeV]",
                     lim=(0,500)),
                Axis(coll="ElectronGood", field="phi",
                     bins=12, start=-pi, stop=pi,
                     label="Electron $\phi$"),
            ]
        ),
        "electron_etaSC_phi_all" : HistConf(
            [
                Axis(coll="ElectronGood", field="phi",
                     bins=12, start=-pi, stop=pi,
                     label="Electron $\phi$"),
                Axis(coll="ElectronGood", field="etaSC", type="variable",
                     bins=[-2.5, -2.0, -1.5660, -1.4442, -1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4442, 1.5660, 2.0, 2.5],
                     label="Electron Supercluster $\eta$",
                     lim=(-2.5,2.5)),
            ]
        ),
    },
)

run_options = {
        "executor"       : "dask/slurm",
        "env"            : "conda",
        "workers"        : 1,
        "scaleout"       : 100,
        "queue"          : "standard",
        "walltime"       : "02:00:00",
        "mem_per_worker" : "4GB", # GB
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
