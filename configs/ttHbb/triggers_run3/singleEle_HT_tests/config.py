from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.cut_definition import Cut
from pocket_coffea.lib.cut_functions import get_nObj_min, get_HLTsel, get_nBtagMin
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.parameters.histograms import *

import workflow
from workflow import TriggerProcessor
    
import cuts
import cloudpickle
cloudpickle.register_pickle_by_value(workflow)
cloudpickle.register_pickle_by_value(cuts)


from cuts import *
import os
localdir = os.path.dirname(os.path.abspath(__file__))

# Loading default parameters
from pocket_coffea.parameters import defaults
default_parameters = defaults.get_default_parameters()
defaults.register_configuration_dir("config_dir", localdir+"/params")

parameters = defaults.merge_parameters_from_files(default_parameters,
                                                  f"{localdir}/params/objects_preselection.yaml",
                                                  f"{localdir}/params/triggers.yaml",
                                                  f"{localdir}/params/general_params.yaml",
                                                  update=True)


cfg = Configurator(
    parameters = parameters,
    datasets = {
        "jsons": [f"{localdir}/datasets/ttHTobb_Run3.json"],
        "filter" : {
            "samples": ["ttHTobb"],
            "samples_exclude" : [],
            "year": ["Run3"]
        }
    },

    workflow = TriggerProcessor,
    workflow_options = {},
    
    skim = [semilep_lhe, semilep_lhe_ele], 
    
    preselections = [],
    categories = {
        "semilep": [passthrough],
        "semilep_Ele28" : [get_HLTsel(["Ele28"])],
        "semilep_Ele30" : [get_HLTsel(["Ele30"])],
        "semilep_Ele32" : [get_HLTsel(["Ele32"])],

        "semilep_tripleB_Ele30": [get_HLTsel(["TripleBTag","Ele30"])] ,
        "semilep_doubleB_tighter_Ele30": [get_HLTsel(["Jet4_btag2_tighter","Ele30"])] ,
        "semilep_doubleB_looser_Ele30": [get_HLTsel(["Jet4_btag2_looser","Ele30"])],
        "semilep_tripleB": [get_HLTsel(["TripleBTag"])] ,
        "semilep_doubleB_tighter": [get_HLTsel(["Jet4_btag2_tighter"])] ,
        "semilep_doubleB_looser": [get_HLTsel(["Jet4_btag2_looser"])],

        # Requesting 4 reconstructed jets with 20 GeV
        "semilep_4j": [ get_nObj_min(4, 20., coll="Jet")],
        "semilep_4j_Ele28" : [get_nObj_min(4, 20., coll="Jet"),  get_HLTsel(["Ele28"])],
        "semilep_4j_Ele30" : [get_nObj_min(4, 20., coll="Jet"),  get_HLTsel(["Ele30"])],
        "semilep_4j_Ele32" : [get_nObj_min(4, 20., coll="Jet"),  get_HLTsel(["Ele32"])],

        "semilep_4j_tripleB_Ele30": [get_nObj_min(4, 20., coll="Jet"), get_HLTsel(["TripleBTag","Ele30"]),] ,
        "semilep_4j_doubleB_tighter_Ele30": [get_nObj_min(4, 20., coll="Jet"),get_HLTsel(["Jet4_btag2_tighter","Ele30"])] ,
        "semilep_4j_doubleB_looser_Ele30": [get_nObj_min(4, 20., coll="Jet"), get_HLTsel(["Jet4_btag2_looser","Ele30"])],
        "semilep_4j_tripleB": [ get_nObj_min(4, 20., coll="Jet"), get_HLTsel(["TripleBTag"])] ,
        "semilep_4j_doubleB_tighter": [get_nObj_min(4, 20., coll="Jet"), get_HLTsel(["Jet4_btag2_tighter"])] ,
        "semilep_4j_doubleB_looser": [get_nObj_min(4, 20., coll="Jet"), get_HLTsel(["Jet4_btag2_looser"])],

        "semilep_recoele15tight": [get_nObj_min(1, coll="ElectronGood")],
        "semilep_recoele15tight_Ele28" : [ get_nObj_min(1, coll="ElectronGood"), get_HLTsel(["Ele28"])],
        "semilep_recoele15tight_Ele30" : [ get_nObj_min(1, coll="ElectronGood"), get_HLTsel(["Ele30"])],
        "semilep_recoele15tight_Ele28_or_Ele30" : [ get_nObj_min(1, coll="ElectronGood"), get_HLTsel(["Ele28","Ele30"])],
        "semilep_recoele15tight_Ele32" : [ get_nObj_min(1, coll="ElectronGood"), get_HLTsel(["Ele32"])],
        "semilep_recoele15tight_tripleB_Ele30": [ get_nObj_min(1, coll="ElectronGood"),
                                      get_HLTsel(["TripleBTag","Ele30"])] ,
        "semilep_recoele15tight_doubleB_tighter_Ele30": [ get_nObj_min(1, coll="ElectronGood"),
                                          get_HLTsel(["Jet4_btag2_tighter","Ele30"])] ,
        "semilep_recoele15tight_doubleB_looser_Ele30": [ get_nObj_min(1, coll="ElectronGood"),
                                         get_HLTsel(["Jet4_btag2_looser","Ele30"])],
        "semilep_recoele15tight_tripleB": [ get_nObj_min(1, coll="ElectronGood"), get_HLTsel(["TripleBTag"])] ,
        "semilep_recoele15tight_doubleB_tighter": [ get_nObj_min(1, coll="ElectronGood"), get_HLTsel(["Jet4_btag2_tighter"])] ,
        "semilep_recoele15tight_doubleB_looser": [ get_nObj_min(1, coll="ElectronGood"), get_HLTsel(["Jet4_btag2_looser"])],

        "semilep_recoele25tight": [get_nObj_min(1, 25., coll="ElectronGood")],
        "semilep_recoele25tight_Ele28" : [ get_nObj_min(1,25., coll="ElectronGood"), get_HLTsel(["Ele28"])],
        "semilep_recoele25tight_Ele30" : [ get_nObj_min(1, 25.,coll="ElectronGood"), get_HLTsel(["Ele30"])],
        "semilep_recoele25tight_Ele28_or_Ele30" : [ get_nObj_min(1,25., coll="ElectronGood"), get_HLTsel(["Ele28","Ele30"])],
        "semilep_recoele25tight_Ele32" : [ get_nObj_min(1,25., coll="ElectronGood"), get_HLTsel(["Ele32"])],
        "semilep_recoele25tight_tripleB_Ele30": [ get_nObj_min(1, 25.,coll="ElectronGood"),
                                      get_HLTsel(["TripleBTag","Ele30"])] ,
        "semilep_recoele25tight_doubleB_tighter_Ele30": [ get_nObj_min(1, 25.,coll="ElectronGood"),
                                          get_HLTsel(["Jet4_btag2_tighter","Ele30"])] ,
        "semilep_recoele25tight_doubleB_looser_Ele30": [ get_nObj_min(1, 25.,coll="ElectronGood"),
                                         get_HLTsel(["Jet4_btag2_looser","Ele30"])],
        "semilep_recoele25tight_tripleB": [ get_nObj_min(1,25., coll="ElectronGood"), get_HLTsel(["TripleBTag"])] ,
        "semilep_recoele25tight_doubleB_tighter": [ get_nObj_min(1, 25.,coll="ElectronGood"), get_HLTsel(["Jet4_btag2_tighter"])] ,
        "semilep_recoele25tight_doubleB_looser": [ get_nObj_min(1, 25.,coll="ElectronGood"), get_HLTsel(["Jet4_btag2_looser"])],

        "semilep_recoele30tight": [get_nObj_min(1, 30., coll="ElectronGood")],
        "semilep_recoele30tight_Ele28" : [ get_nObj_min(1,30., coll="ElectronGood"), get_HLTsel(["Ele28"])],
        "semilep_recoele30tight_Ele30" : [ get_nObj_min(1, 30.,coll="ElectronGood"), get_HLTsel(["Ele30"])],
        "semilep_recoele30tight_Ele28_or_Ele30" : [ get_nObj_min(1,30., coll="ElectronGood"), get_HLTsel(["Ele28","Ele30"])],
        "semilep_recoele30tight_Ele32" : [ get_nObj_min(1,30., coll="ElectronGood"), get_HLTsel(["Ele32"])],
        "semilep_recoele30tight_tripleB_Ele30": [ get_nObj_min(1, 30.,coll="ElectronGood"),
                                      get_HLTsel(["TripleBTag","Ele30"])] ,
        "semilep_recoele30tight_doubleB_tighter_Ele30": [ get_nObj_min(1, 30.,coll="ElectronGood"),
                                          get_HLTsel(["Jet4_btag2_tighter","Ele30"])] ,
        "semilep_recoele30tight_doubleB_looser_Ele30": [ get_nObj_min(1, 30.,coll="ElectronGood"),
                                         get_HLTsel(["Jet4_btag2_looser","Ele30"])],
        "semilep_recoele30tight_tripleB": [ get_nObj_min(1,30., coll="ElectronGood"), get_HLTsel(["TripleBTag"])] ,
        "semilep_recoele30tight_doubleB_tighter": [ get_nObj_min(1, 30.,coll="ElectronGood"), get_HLTsel(["Jet4_btag2_tighter"])] ,
        "semilep_recoele30tight_doubleB_looser": [ get_nObj_min(1, 30.,coll="ElectronGood"), get_HLTsel(["Jet4_btag2_looser"])],


        "semilep_recoele30tight3b": [get_nObj_min(1, 30., coll="ElectronGood"), get_nBtagMin(3, coll="Jet", minpt=30.)],
        "semilep_recoele30tight3b_Ele28" : [ get_nObj_min(1,30., coll="ElectronGood"), get_nBtagMin(3, coll="Jet", minpt=30.), get_HLTsel(["Ele28"])],
        "semilep_recoele30tight3b_Ele30" : [ get_nObj_min(1, 30.,coll="ElectronGood"), get_nBtagMin(3, coll="Jet", minpt=30.), get_HLTsel(["Ele30"])],
        "semilep_recoele30tight3b_Ele28_or_Ele30" : [ get_nObj_min(1,30., coll="ElectronGood"), get_nBtagMin(3, coll="Jet", minpt=30.), get_HLTsel(["Ele28","Ele30"])],
        "semilep_recoele30tight3b_Ele32" : [ get_nObj_min(1,30., coll="ElectronGood"), get_nBtagMin(3, coll="Jet", minpt=30.), get_HLTsel(["Ele32"])],
        "semilep_recoele30tight3b_tripleB_Ele30": [ get_nObj_min(1, 30.,coll="ElectronGood"), get_nBtagMin(3, coll="Jet", minpt=30.),
                                        get_HLTsel(["TripleBTag","Ele30"])] ,
        "semilep_recoele30tight3b_doubleB_tighter_Ele30": [ get_nObj_min(1, 30.,coll="ElectronGood"), get_nBtagMin(3, coll="Jet", minpt=30.),
                                            get_HLTsel(["Jet4_btag2_tighter","Ele30"])] ,
        "semilep_recoele30tight3b_doubleB_looser_Ele30": [ get_nObj_min(1, 30.,coll="ElectronGood"), get_nBtagMin(3, coll="Jet", minpt=30.),
                                             get_HLTsel(["Jet4_btag2_looser","Ele30"])],
        "semilep_recoele30tight3b_tripleB": [ get_nObj_min(1,30., coll="ElectronGood"), get_nBtagMin(3, coll="Jet", minpt=30.), get_HLTsel(["TripleBTag"])] ,
        "semilep_recoele30tight3b_doubleB_tighter": [ get_nObj_min(1, 30.,coll="ElectronGood"), get_nBtagMin(3, coll="Jet", minpt=30.), get_HLTsel(["Jet4_btag2_tighter"])] ,
        "semilep_recoele30tight3b_doubleB_looser": [ get_nObj_min(1, 30.,coll="ElectronGood"), get_nBtagMin(3, coll="Jet", minpt=30.), get_HLTsel(["Jet4_btag2_looser"])],
        
    },
    
    weights= {
        "common": {
            "inclusive": [
                "genWeight", "lumi","XS",
            ],
            "bycategory": {},
        },
        "bysample": {},
    },
    variations = {
        "weights": {"common": {"inclusive": [], "bycategory": {}},
                    "bysample": {}},
    },

    variables = {
        "higgs_pt": HistConf([
            Axis(coll="higgs", field="pt",
                 bins=30, start=0, stop=400, label="Higgs $p_T$")
        ]),
        "HT": HistConf([
            Axis(coll="LHE", field="HT", 
                 bins=40, start=0, stop=1500, label="LHE $H_T$")
        ]),
        "recoHT": HistConf([
            Axis(coll="events", field="recoHT", 
                 bins=40, start=0, stop=1500, label="reco $H_T$")
        ]),
        "Electron_pt":  HistConf([
            Axis(coll="Electron", field="pt", pos=0,
                 bins=50, start=0, stop=400, label="Electron $p_T$")
        ]),
        "ElectronGood_pt":  HistConf([
            Axis(coll="ElectronGood", field="pt", pos=0,
                 bins=50, start=0, stop=400, label="Electron $p_T$")
        ]),
        "ElectronGood_pt2":  HistConf([
            Axis(coll="ElectronGood", field="pt", pos=0,
                 bins=50, start=10, stop=150, label="Electron $p_T$")
        ]),
        "ElectronGood_pt3":  HistConf([
            Axis(coll="ElectronGood", field="pt", pos=0,
                 bins=50, start=0, stop=100, label="Electron $p_T$")
        ]),
        "ElectronGood_pt4":  HistConf([
            Axis(coll="ElectronGood", field="pt", pos=0,
                 bins=50, start=15, stop=50, label="Electron $p_T$")
        ]),
        **count_hist(name="nJets", coll="Jets", bins=12, start=2, stop=14),

    
    },

)


  

    
