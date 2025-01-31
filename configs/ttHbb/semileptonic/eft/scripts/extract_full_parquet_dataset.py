import numpy as np
import pandas as pd
import awkward as ak
#from numba import njit
import vector
vector.register_numba()
vector.register_awkward()

import os
import argparse

args = argparse.ArgumentParser()
args.add_argument("--name", type=str, required=True)
args.add_argument("--input", type=str, required=True)
args.add_argument("--output", type=str, required=True)
args.add_argument("--only-datasets", type=str, nargs="+", default=None)
args.add_argument("--only-categories", type=str, nargs="+", default=None)
args = args.parse_args()

os.makedirs(args.output, exist_ok=True)
basedir= args.input
datasets =os.listdir(basedir)

pq_datasets = []
for d in datasets:
    if args.only_datasets is not None:
        if d not in args.only_datasets:
            continue
    categories = os.listdir(basedir + f"/{d}")
    if args.only_categories is not None:
        categories = [c for c in categories if c in args.only_categories]
    for c in categories:
        # create the overall index of the parquet dataset
        ak.to_parquet.dataset(basedir + f"/{d}/{c}")
        pq_datasets.append(basedir + f"/{d}/{c}")


# Now we read back the parquet dataset and extract the info we need in a single parquet file
for dataset in pq_datasets:
    print(f"Preparing {dataset}")
    cs = ak.from_parquet(dataset, use_threads=4)
    
    partons_initial = ak.zip({"pt": cs["PartonInitial_pt"],
                    "eta": cs["PartonInitial_eta"],
                    "phi": cs["PartonInitial_phi"],
                     "mass": cs["PartonInitial_mass"],
                    "pdgId": cs["PartonInitial_pdgId"], 
                    "prov": cs["PartonInitial_provenance"]},
                                 with_name='Momentum4D')

    partons_lastcopy = ak.zip({"pt": cs["PartonLastCopy_pt"],
                    "eta": cs["PartonLastCopy_eta"],
                    "phi": cs["PartonLastCopy_phi"],
                     "mass": cs["PartonLastCopy_mass"],
                    "pdgId": cs["PartonLastCopy_pdgId"], 
                    "prov": cs["PartonLastCopy_provenance"]},
                                 with_name='Momentum4D')
    
    partons_matched = ak.zip({"pt": cs["PartonInitialMatched_pt"],
                                  "eta": cs["PartonInitialMatched_eta"],
                                  "phi": cs["PartonInitialMatched_phi"],
                                  "mass": cs["PartonInitialMatched_mass"],
                                  "pdgId": cs["PartonInitialMatched_pdgId"], 
                                  "prov": cs["PartonInitialMatched_provenance"]},
                             with_name='Momentum4D')

    jets = ak.zip({"pt": cs["JetGood_pt"],
                                  "eta": cs["JetGood_eta"],
                                  "phi": cs["JetGood_phi"],
                                  "btag": cs["JetGood_btagDeepFlavB"],
                                  "m": ak.zeros_like(cs["JetGood_btagDeepFlavB"])},
                             with_name='Momentum4D')

    jets_matched = ak.zip({"pt": cs["JetGoodMatched_pt"],
                                  "eta": cs["JetGoodMatched_eta"],
                                  "phi": cs["JetGoodMatched_phi"],
                                  "btag": cs["JetGoodMatched_btagDeepFlavB"],
                                  "prov": cs["PartonLastCopyMatched_provenance"],
                                  "m": ak.zeros_like(cs["PartonLastCopyMatched_provenance"])},
                             with_name='Momentum4D')


    generator_info = ak.zip({"pdgid1": cs["Generator_id1"],
                                  "pdgid2": cs["Generator_id2"],
                                  "x1": cs["Generator_x1"],
                                  "x2": cs["Generator_x2"]},
                             )


    lepton_gen = ak.zip({"pt": cs["LeptonGenLevel_pt"],
                                  "eta": cs["LeptonGenLevel_eta"],
                                  "phi": cs["LeptonGenLevel_phi"],
                                  "mass": cs["LeptonGenLevel_mass"],
                                  "pdgId": cs["LeptonGenLevel_pdgId"]},
                             with_name='Momentum4D')


    lepton_reco = ak.zip({"pt": cs["LeptonGood_pt"],
                                  "eta": cs["LeptonGood_eta"],
                                  "phi": cs["LeptonGood_phi"],
                                  "m": ak.zeros_like(cs["LeptonGood_pt"])},
                             with_name='Momentum4D')


    met = ak.zip({"pt": cs["MET_pt"],
                  "eta":  ak.zeros_like(cs["MET_pt"]),
                  "phi": cs["MET_phi"],
                  "m": ak.zeros_like(cs["MET_pt"])},
             with_name='Momentum4D')

    higgs = ak.zip({"pt": cs["HiggsGen_pt"],
                    "eta": cs["HiggsGen_eta"],
                    "phi": cs["HiggsGen_phi"],
                    "m": cs["HiggsGen_mass"]},
                   with_name='Momentum4D')


    top = ak.zip({"pt": cs["TopGen_pt"],
                  "eta": cs["TopGen_eta"],
                    "phi": cs["TopGen_phi"],
                    "mass": cs["TopGen_mass"],
                    "pdgId": cs["TopGen_pdgId"]},
                    with_name='Momentum4D')
    
    antitop = ak.zip({"pt": cs["AntiTopGen_pt"],
                    "eta": cs["AntiTopGen_eta"],
                        "phi": cs["AntiTopGen_phi"],
                        "mass": cs["AntiTopGen_mass"],
                        "pdgId": cs["AntiTopGen_pdgId"]},
                        with_name='Momentum4D')

    # isr = ak.zip({"pt": cs["ISR_pt"],
    #                 "eta": cs["ISR_eta"],
    #                     "phi": cs["ISR_phi"],
    #                     "mass": cs["ISR_mass"],
    #                     "pdgId": cs["ISR_pdgId"]},
    #                     with_name='Momentum4D')

    jets_matched = ak.mask(jets_matched, jets_matched.pt==-999, None)
    partons_matched = ak.mask(partons_matched, partons_matched.pt==-999, None)
    is_jet_matched = ~ak.is_none(jets_matched, axis=1)
    jets = ak.with_field(jets, is_jet_matched, "matched")

    # Filling with -1 the not matched provenance
    jets = ak.with_field(jets, ak.fill_none(jets_matched.prov, -1), "prov")
    
    dfout = ak.zip({
            "jets": jets,
            "partons_matched": partons_matched,
            "partons_initial": partons_initial,
            "partons_lastcopy": partons_lastcopy,
            "generator_info": generator_info,
            "lepton_gen":lepton_partons,
            "lepton_reco": lepton_reco,
            "met": met,
            "higgs": higgs,
            "top": top,
            "antitop": antitop,
          #  "isr": isr,
            "weight": cs["weight"]
            }, depth_limit=1)


    ak.to_parquet(dfout, f"{args.out}/{dataset}_{name}.parquet")
