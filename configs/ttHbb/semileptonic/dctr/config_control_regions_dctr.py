from pocket_coffea.utils.configurator import Configurator
from pocket_coffea.lib.cut_definition import Cut
from pocket_coffea.lib.columns_manager import ColOut
from pocket_coffea.lib.cut_functions import get_nObj_eq, get_nObj_min, get_HLTsel, get_nBtagMin, get_nPVgood, goldenJson, eventFlags
from pocket_coffea.lib.weights.common.common import common_weights
from pocket_coffea.lib.weights.common.weights_run2_UL import SF_ele_trigger, SF_QCD_renorm_scale, SF_QCD_factor_scale
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.parameters.histograms import *

import configs
import configs.ttHbb.semileptonic.common.workflows.workflow_dctr as workflow
from configs.ttHbb.semileptonic.common.workflows.workflow_dctr import DCTRInferenceProcessor
from configs.ttHbb.semileptonic.common.executors import onnx_executor as onnx_executor
import configs.ttHbb.semileptonic.common.params.quantile_transformer as quantile_transformer
from configs.ttHbb.semileptonic.common.params.quantile_transformer import WeightedQuantileTransformer

import configs.ttHbb.semileptonic.common.cuts.custom_cut_functions as custom_cut_functions
import configs.ttHbb.semileptonic.common.cuts.custom_cuts as custom_cuts
from configs.ttHbb.semileptonic.common.cuts.custom_cut_functions import *
from configs.ttHbb.semileptonic.common.cuts.custom_cuts import *
from configs.ttHbb.semileptonic.common.weights.custom_weights import SF_top_pt, SF_LHE_pdf_weight
from params.axis_settings import axis_settings

import os
import json
localdir = os.path.dirname(os.path.abspath(__file__))

# Define tthbb working points for SPANet
tthbb_L = 0.4
tthbb_M = 0.75
ttlf_wp = 0.3

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
                                                  f"{localdir}/params/plotting_style_dctr.yaml",
                                                  f"{localdir}/params/ml_models_T3_CH_PSI.yaml",
                                                  f"{localdir}/params/quantile_transformer.yaml",
                                                  update=True)

samples = ["ttHTobb",
           "ttHTobb_ttToSemiLep",
           "TTbbSemiLeptonic",
           "TTToSemiLeptonic",
           "TTTo2L2Nu",
           "SingleTop",
           "WJetsToLNu_HT",
           "DYJetsToLL",
           "VV",
           "TTV",
           "DATA_SingleEle",
           "DATA_SingleMuon"
           ]
samples_with_qcd = [s for s in samples if s not in ["VV", "DATA_SingleEle", "DATA_SingleMuon"]]

with open(parameters["dctr"]["weight_cuts"]["by_njet"]["file"]) as f:
    w_cuts = json.load(f)

# Set the limit of the last quantile for each key as inf
for key in w_cuts.keys():
    w_cuts[key][2][1] = float("inf")

with open(parameters["dctr"]["weight_cuts"]["inclusive"]["file"]) as f:
    w_cuts_inclusive = json.load(f)["weight_cuts"]["quantile0p33"]

cfg = Configurator(
    parameters = parameters,
    datasets = {
        "jsons": [f"{localdir}/datasets/datasets_Run2_skim.json",
                  ],
        "filter" : {
            "samples": samples,
            "samples_exclude" : [],
            "year": ["2016_PreVFP",
                     "2016_PostVFP",
                     "2017",
                     "2018"
                     ] #All the years
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
                'TTbbSemiLeptonic_tt+C'    : [get_genTtbarId_100_eq([41, 42, 43, 44, 45, 46])],
                'TTbbSemiLeptonic_tt+B'    : [get_genTtbarId_100_eq([51, 52, 53, 54, 55, 56])],
                'TTbbSemiLeptonic_tt+B_4j_DCTR_L'    : [get_genTtbarId_100_eq([51, 52, 53, 54, 55, 56]), get_nObj_eq(4, coll="JetGood"), get_w_dctr_interval(*w_cuts["njet=4"][0])],
                'TTbbSemiLeptonic_tt+B_4j_DCTR_M'    : [get_genTtbarId_100_eq([51, 52, 53, 54, 55, 56]), get_nObj_eq(4, coll="JetGood"), get_w_dctr_interval(*w_cuts["njet=4"][1])],
                'TTbbSemiLeptonic_tt+B_4j_DCTR_H'    : [get_genTtbarId_100_eq([51, 52, 53, 54, 55, 56]), get_nObj_eq(4, coll="JetGood"), get_w_dctr_interval(*w_cuts["njet=4"][2])],
                'TTbbSemiLeptonic_tt+B_5j_DCTR_L'    : [get_genTtbarId_100_eq([51, 52, 53, 54, 55, 56]), get_nObj_eq(5, coll="JetGood"), get_w_dctr_interval(*w_cuts["njet=5"][0])],
                'TTbbSemiLeptonic_tt+B_5j_DCTR_M'    : [get_genTtbarId_100_eq([51, 52, 53, 54, 55, 56]), get_nObj_eq(5, coll="JetGood"), get_w_dctr_interval(*w_cuts["njet=5"][1])],
                'TTbbSemiLeptonic_tt+B_5j_DCTR_H'    : [get_genTtbarId_100_eq([51, 52, 53, 54, 55, 56]), get_nObj_eq(5, coll="JetGood"), get_w_dctr_interval(*w_cuts["njet=5"][2])],
                'TTbbSemiLeptonic_tt+B_6j_DCTR_L'    : [get_genTtbarId_100_eq([51, 52, 53, 54, 55, 56]), get_nObj_eq(6, coll="JetGood"), get_w_dctr_interval(*w_cuts["njet=6"][0])],
                'TTbbSemiLeptonic_tt+B_6j_DCTR_M'    : [get_genTtbarId_100_eq([51, 52, 53, 54, 55, 56]), get_nObj_eq(6, coll="JetGood"), get_w_dctr_interval(*w_cuts["njet=6"][1])],
                'TTbbSemiLeptonic_tt+B_6j_DCTR_H'    : [get_genTtbarId_100_eq([51, 52, 53, 54, 55, 56]), get_nObj_eq(6, coll="JetGood"), get_w_dctr_interval(*w_cuts["njet=6"][2])],
                'TTbbSemiLeptonic_tt+B_>=7j_DCTR_L'    : [get_genTtbarId_100_eq([51, 52, 53, 54, 55, 56]), get_nObj_min(7, coll="JetGood"), get_w_dctr_interval(*w_cuts["njet>=7"][0])],
                'TTbbSemiLeptonic_tt+B_>=7j_DCTR_M'    : [get_genTtbarId_100_eq([51, 52, 53, 54, 55, 56]), get_nObj_min(7, coll="JetGood"), get_w_dctr_interval(*w_cuts["njet>=7"][1])],
                'TTbbSemiLeptonic_tt+B_>=7j_DCTR_H'    : [get_genTtbarId_100_eq([51, 52, 53, 54, 55, 56]), get_nObj_min(7, coll="JetGood"), get_w_dctr_interval(*w_cuts["njet>=7"][2])],
            },
            'TTToSemiLeptonic' : {
                #'TTToSemiLeptonic_tt+LF'   : [get_genTtbarId_100_eq(0)],
                'TTToSemiLeptonic_tt+C'    : [get_genTtbarId_100_eq([41, 42, 43, 44, 45, 46])],
                'TTToSemiLeptonic_tt+B'    : [get_genTtbarId_100_eq([51, 52, 53, 54, 55, 56])],
            },
        }
    },

    workflow = DCTRInferenceProcessor,
    workflow_options = {"parton_jet_min_dR": 0.3,
                        "spanet_model": parameters["spanet"]["file"],
                        "dctr_model": parameters["dctr"]["file"],
                        },
    
    skim = [get_nPVgood(1),
            eventFlags,
            goldenJson,
            get_nObj_min(4, 15., "Jet"),
            get_nBtagMin(3, 15., coll="Jet", wp="M"),
            get_HLTsel(primaryDatasets=["SingleEle", "SingleMuon"])],
    
    preselections = [semileptonic_presel],
    categories = {
        "semilep": [passthrough],
        f"ttlf0p{int(100*ttlf_wp)}": [get_ttlf_max(ttlf_wp)],
        "CR_ttlf": [get_ttlf_min(ttlf_wp)],
        f"CR_tthbb0p00To0p{int(100*tthbb_L)}": [get_ttlf_max(ttlf_wp), get_CR(0., tthbb_L)],
        "CR": [get_ttlf_max(ttlf_wp), get_CR(tthbb_L, tthbb_M)],
        "SR": [get_ttlf_max(ttlf_wp), get_SR(tthbb_M)],
        f"4jCR_tthbb0p00To0p{int(100*tthbb_L)}": [get_ttlf_max(ttlf_wp), get_CR(0., tthbb_L), get_nObj_eq(4, coll="JetGood")],
        "4jCR": [get_ttlf_max(ttlf_wp), get_CR(tthbb_L, tthbb_M), get_nObj_eq(4, coll="JetGood")],
        "4jSR": [get_ttlf_max(ttlf_wp), get_SR(tthbb_M), get_nObj_eq(4, coll="JetGood")],
        f"5jCR_tthbb0p00To0p{int(100*tthbb_L)}": [get_ttlf_max(ttlf_wp), get_CR(0., tthbb_L), get_nObj_eq(5, coll="JetGood")],
        "5jCR": [get_ttlf_max(ttlf_wp), get_CR(tthbb_L, tthbb_M), get_nObj_eq(5, coll="JetGood")],
        "5jSR": [get_ttlf_max(ttlf_wp), get_SR(tthbb_M), get_nObj_eq(5, coll="JetGood")],
        f"6jCR_tthbb0p00To0p{int(100*tthbb_L)}": [get_ttlf_max(ttlf_wp), get_CR(0., tthbb_L), get_nObj_eq(6, coll="JetGood")],
        "6jCR": [get_ttlf_max(ttlf_wp), get_CR(tthbb_L, tthbb_M), get_nObj_eq(6, coll="JetGood")],
        "6jSR": [get_ttlf_max(ttlf_wp), get_SR(tthbb_M), get_nObj_eq(6, coll="JetGood")],
        f">=7jCR_tthbb0p00To0p{int(100*tthbb_L)}": [get_ttlf_max(ttlf_wp), get_CR(0., tthbb_L), get_nObj_min(7, coll="JetGood")],
        ">=7jCR": [get_ttlf_max(ttlf_wp), get_CR(tthbb_L, tthbb_M), get_nObj_min(7, coll="JetGood")],
        ">=7jSR": [get_ttlf_max(ttlf_wp), get_SR(tthbb_M), get_nObj_min(7, coll="JetGood")],
    },

    weights_classes = common_weights + [SF_ele_trigger, SF_top_pt, SF_QCD_renorm_scale, SF_QCD_factor_scale, SF_LHE_pdf_weight],
    weights= {
        "common": {
            "inclusive": [
                "genWeight", "lumi","XS",
                "pileup",
                "sf_ele_reco", "sf_ele_id", "sf_ele_trigger",
                "sf_mu_id", "sf_mu_iso", "sf_mu_trigger",
                "sf_btag", "sf_btag_calib",
                "sf_jet_puId", "sf_top_pt",
                "sf_partonshower_isr", "sf_partonshower_fsr",
            ],
            "bycategory": {},
        },
        "bysample": {
            s : { "inclusive": ["sf_qcd_renorm_scale", "sf_qcd_factor_scale", "sf_lhe_pdf_weight"] } for s in samples_with_qcd
        },
    },
    variations = {
        "weights": {
            "common": {
                "inclusive": ["pileup",
                              "sf_ele_reco", "sf_ele_id", "sf_ele_trigger",
                              "sf_mu_id", "sf_mu_iso", "sf_mu_trigger",
                              "sf_btag", "sf_btag_calib",
                              "sf_jet_puId", "sf_top_pt",
                              "sf_partonshower_isr", "sf_partonshower_fsr",
                              ],
                "bycategory": {}
            },
            "bysample": {
                s : { "inclusive": ["sf_qcd_renorm_scale", "sf_qcd_factor_scale", "sf_lhe_pdf_weight"] } for s in samples_with_qcd
            },
        },
        "shape": {
            "common": {
                "inclusive" : ["JES_Total_AK4PFchs", "JER_AK4PFchs"]
            }
        }
    },
    
    variables = {
        **count_hist(name="nLeptons", coll="LeptonGood",bins=3, start=0, stop=3),
        **count_hist(name="nJets", coll="JetGood",bins=10, start=4, stop=14),
        **count_hist(name="nBJets", coll="BJetGood",bins=10, start=0, stop=10),
        **ele_hists(axis_settings=axis_settings),
        **muon_hists(axis_settings=axis_settings),
        **met_hists(coll="MET", axis_settings=axis_settings),
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
        "jets_Ht" : HistConf(
          [Axis(coll="events", field="JetGood_Ht", bins=25, start=0, stop=2500,
                label="Jets $H_T$ [GeV]")]
        ),
        "jets_Ht_coarsebins" : HistConf(
          [Axis(coll="events", field="JetGood_Ht", type="variable", bins=[0,100,200,300,400,500,600,700,800,900,1000,1250,1500,2000,2500], start=0, stop=2500,
                label="Jets $H_T$ [GeV]")]
        ),
        "jets_Ht_coarsebins2" : HistConf(
          [Axis(coll="events", field="JetGood_Ht", type="variable", bins=[0,100,200,300,400,500,600,700,800,900,1000,1250,1500], start=0, stop=1500,
                label="Jets $H_T$ [GeV]")]
        ),
        "jets_Ht_finerbins" : HistConf(
          [Axis(coll="events", field="JetGood_Ht", type="variable",
                bins=[0,50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000,1250,1500,2000,2500], start=0, stop=2500,
                label="Jets $H_T$ [GeV]")]
        ),
        "bjets_Ht" : HistConf(
          [Axis(coll="events", field="BJetGood_Ht", bins=25, start=0, stop=2500,
                label="B-Jets $H_T$ [GeV]")]
        ),
        "lightjets_Ht" : HistConf(
          [Axis(coll="events", field="LightJetGood_Ht", bins=25, start=0, stop=2500,
                label="Light-Jets $H_T$ [GeV]")]
        ),
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
        "mbb_closest" : HistConf(
            [Axis(coll="events", field="mbb_closest", bins=50, start=0, stop=500,
                    label="$m_{bb}(min \Delta R(bb))$ [GeV]")]
        ),
        "mbb_min" : HistConf(
            [Axis(coll="events", field="mbb_min", bins=50, start=0, stop=500,
                    label="$m_{bb}^{min}$ [GeV]")]
        ),
        "mbb_max" : HistConf(
            [Axis(coll="events", field="mbb_max", bins=50, start=0, stop=500,
                    label="$m_{bb}^{max}$ [GeV]")]
        ),
        "deltaRbb_avg" : HistConf(
            [Axis(coll="events", field="deltaRbb_avg", bins=50, start=0, stop=5,
                  label="$\Delta R_{bb}^{avg}$")]
        ),
        "ptbb_closest" : HistConf(
            [Axis(coll="events", field="ptbb_closest", bins=axis_settings["jet_pt"]["bins"], start=axis_settings["jet_pt"]["start"], stop=axis_settings["jet_pt"]["stop"],
                    label="$p_{T,bb}(min \Delta R(bb))$ [GeV]")]
        ),
        "htbb_closest" : HistConf(
            [Axis(coll="events", field="htbb_closest", bins=25, start=0, stop=2500,
                    label="$H_{T,bb}(min \Delta R(bb))$ [GeV]")]
        ),
        "spanet_tthbb" : HistConf(
            [Axis(coll="spanet_output", field="tthbb", bins=50, start=0, stop=1, label="tthbb SPANet score")],
        ),
        "spanet_tthbb_transformed" : HistConf(
            [Axis(coll="spanet_output", field="tthbb_transformed", bins=50, start=0, stop=1, label="tthbb SPANet transformed score")],
        ),
        "spanet_tthbb_transformed_binning0p025" : HistConf(
            [Axis(coll="spanet_output", field="tthbb_transformed", bins=40, start=0, stop=1, label="tthbb SPANet transformed score")],
        ),
        "spanet_tthbb_transformed_binning0p04" : HistConf(
            [Axis(coll="spanet_output", field="tthbb_transformed", bins=25, start=0, stop=1, label="tthbb SPANet transformed score")],
        ),
        "spanet_tthbb_transformed_binning0p05" : HistConf(
            [Axis(coll="spanet_output", field="tthbb_transformed", bins=20, start=0, stop=1, label="tthbb SPANet transformed score")],
        ),
        "spanet_ttbb" : HistConf(
            [Axis(coll="spanet_output", field="ttbb", bins=50, start=0, stop=1, label="ttbb SPANet score")],
        ),
        "spanet_ttcc" : HistConf(
            [Axis(coll="spanet_output", field="ttcc", bins=50, start=0, stop=1, label="ttcc SPANet score")],
        ),
        "spanet_ttlf" : HistConf(
            [Axis(coll="spanet_output", field="ttlf", bins=50, start=0, stop=1, label="ttlf SPANet score")],
        ),
        "dctr_score" : HistConf(
            [Axis(coll="dctr_output", field="score", bins=50, start=0, stop=1, label="DCTR score")],
        ),
        "dctr_weight" : HistConf(
            [Axis(coll="dctr_output", field="weight", bins=50, start=0, stop=2.5, label="DCTR weight")],
        ),
        "dctr_index" : HistConf(
            [Axis(coll="dctr_output", field="index", bins=13, start=0, stop=13, label="DCTR index")],
        ),
    },
)

# Registering custom functions
import cloudpickle
cloudpickle.register_pickle_by_value(workflow)
cloudpickle.register_pickle_by_value(custom_cut_functions)
cloudpickle.register_pickle_by_value(custom_cuts)
cloudpickle.register_pickle_by_value(quantile_transformer)
cloudpickle.register_pickle_by_value(onnx_executor)
cloudpickle.register_pickle_by_value(configs)
