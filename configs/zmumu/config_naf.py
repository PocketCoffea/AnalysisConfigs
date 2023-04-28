import awkward as ak
from pocket_coffea.parameters.cuts.preselection_cuts import passthrough
from pocket_coffea.lib.cut_definition import Cut
from pocket_coffea.lib.cut_functions import get_nObj_min, get_HLTsel
from pocket_coffea.parameters.histograms import *
from configs.zmumu.workflow import zmumuBaseProcessor
from configs.zmumu import datasets
datasets_abspath = datasets.__path__[0]

trigger_dict = {
    "2018": {
        "SingleEle": [
            "Ele32_WPTight_Gsf",
            "Ele28_eta2p1_WPTight_Gsf_HT150",
        ],
        "SingleMuon": [
            "IsoMu24",
        ],
    },
}

def dimuon(events, params, year, sample, **kwargs):

    # Masks for same-flavor (SF) and opposite-sign (OS)
    SF = ((events.nMuonGood == 2) & (events.nElectronGood == 0))
    OS = events.ll.charge == 0

    mask = (
        (events.nLeptonGood == 2)
        & (ak.firsts(events.MuonGood.pt) > params["pt_leading_muon"])
        & OS & SF
        & (events.ll.mass > params["mll"]["low"])
        & (events.ll.mass < params["mll"]["high"])
    )

    # Pad None values with False
    return ak.where(ak.is_none(mask), False, mask)

dimuon_presel = Cut(
    name="dilepton",
    params={
        "pt_leading_muon": 25,
        "mll": {'low': 25, 'high': 2000},
    },
    function=dimuon,
)

cfg =  {
    "dataset" : {
        "jsons": [f"{datasets_abspath}/DATA_SingleMuonC_redirector.json",
                  f"{datasets_abspath}/DYJetsToLL_M-50.json"
                 ],
        "filter" : {
            "samples": ["DATA_SingleMuonC",
                        "DYJetsToLL"],
            "samples_exclude" : [],
            "year": ['2018']
        },
    },
    

    # Input and output files
    "workflow" : zmumuBaseProcessor,
    "output"   : "output/test_zmumu",
    "workflow_options" : {},

    "run_options" : {
        "executor"       : "parsl/condor/naf-desy",
        "workers"        : 1,
        "scaleout"       : 300,
        "queue"          : "microcentury",
        "walltime"       : "00:40:00",
        "disk_per_worker": "4GB",
        "mem_per_worker" : "2GB", # GB
        "exclusive"      : False,
        "chunk"          : 200000,
        "retries"        : 20,
        # "treereduction"  : None,
        "max"            : None,
        "skipbadfiles"   : None,
        "voms"           : None,
        "limit"          : None,
        "adapt"          : False,
        
    },

    # Cuts and plots settings
    "finalstate" : "dimuon",
    "skim": [get_nObj_min(1, 15., "Muon"),
             get_HLTsel("dimuon", trigger_dict, primaryDatasets=["SingleMuon"])],
    "preselections" : [dimuon_presel],
    "categories": {
        "baseline": [passthrough],
    },

    "weights": {
        "common": {
            "inclusive": ["genWeight","lumi","XS",
                          "pileup",
                          "sf_mu_id","sf_mu_iso",
                          ],
            "bycategory" : {
            }
        },
        "bysample": {
        }
    },

    "variations": {
        "weights": {
            "common": {
                "inclusive": [  "pileup",
                                "sf_mu_id", "sf_mu_iso"
                              ],
                "bycategory" : {
                }
            },
        "bysample": {
        }    
        },
        
    },

   "variables":
    {
        **muon_hists(coll="MuonGood", pos=0),
        **count_hist(name="nElectronGood", coll="ElectronGood",bins=3, start=0, stop=3),
        **count_hist(name="nMuonGood", coll="MuonGood",bins=3, start=0, stop=3),
        **count_hist(name="nJets", coll="JetGood",bins=8, start=0, stop=8),
        **count_hist(name="nBJets", coll="BJetGood",bins=8, start=0, stop=8),
        **jet_hists(coll="JetGood", pos=0),
        **jet_hists(coll="JetGood", pos=1),
        "mll" : HistConf( [Axis(coll="ll", field="mass", bins=100, start=50, stop=150, label=r"$M_{\ell\ell}$ [GeV]")] ),

    }
}
