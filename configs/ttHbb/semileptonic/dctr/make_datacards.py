import os
import argparse
from coffea.util import load
from pocket_coffea.utils.datacard import Datacard, combine_datacards
from pocket_coffea.utils.processes import Process
from pocket_coffea.utils.systematics import SystematicUncertainty

parser = argparse.ArgumentParser(description="Make datacards")
parser.add_argument("-i", "--input", help="Coffea input file with histograms", required=True)
parser.add_argument("-o", "--output", help="Output directory for datacards", default="datacards", required=False)
parser.add_argument("--year", help="Year of the datacards", required=True)
args = parser.parse_args()

df = load(args.input)
datasets_metadata = df["datasets_metadata"]

samples_data = [s for s in datasets_metadata["by_datataking_period"][args.year].keys() if s.startswith("DATA")]

processes = [
    Process(name="tthbb", samples=["ttHTobb"], is_signal=True),
    Process(name="ttlf", samples=["TTToSemiLeptonic__TTToSemiLeptonic_tt+LF"], is_signal=False),
    Process(name="ttcc", samples=["TTToSemiLeptonic__TTToSemiLeptonic_tt+C"], is_signal=False),
    Process(name="ttbb", samples=["TTbbSemiLeptonic__TTbbSemiLeptonic_tt+B"], is_signal=False),
    Process(name="tt_dilepton", samples=["TTTo2L2Nu"], is_signal=False),
    Process(name="singletop", samples=["SingleTop"], is_signal=False),
    Process(name="vjets", samples=["WJetsToLNu_HT", "DYJetsToLL"], is_signal=False),
    Process(name="ttv", samples=["TTV"], is_signal=False),
    Process(name="diboson", samples=["VV"], is_signal=False),
]

data_processes = [Process(name="data_obs", samples=samples_data, is_signal=False, is_data=True)]

systematics = [
    SystematicUncertainty(name="lumi", typ="lnN", processes=[p.name for p in processes if p not in samples_data], value=1.025),
]
shape_systematics = [
    "pileup",
    "sf_ele_reco", "sf_ele_id",
    "sf_ele_trigger_era", "sf_ele_trigger_ht",
    "sf_ele_trigger_pileup", "sf_ele_trigger_stat",
    "sf_mu_id", "sf_mu_iso", "sf_mu_trigger",
    "sf_btag_cferr1", "sf_btag_cferr2",
    "sf_btag_hf", "sf_btag_hfstats1", "sf_btag_hfstats2",
    "sf_btag_lf", "sf_btag_lfstats1", "sf_btag_lfstats2",
    "sf_jet_puId", #"sf_top_pt",
    #"sf_btag_calib", "sf_ttlf_calib",
]
for syst in shape_systematics:
    systematics.append(SystematicUncertainty(name=syst, typ="shape", processes=[p.name for p in processes if p not in samples_data], value=1.0))

print("Systematic lnN uncertainties: ", [s.name for s in systematics if s.typ == "lnN"])
print("Systematic shape uncertainties: ", [s.name for s in systematics if s.typ == "shape"])

histograms_dict = {
    "CR" : df["variables"]["dctr_index"],
    "CR_ttlf" : df["variables"]["jets_Ht"],
    "SR" : df["variables"]["spanet_tthbb_transformed"],
}

datacards = {}
for cat, histograms in histograms_dict.items():
    print(f"Creating datacard for category: {cat}")
    datacard = Datacard(
        histograms=histograms,
        datasets_metadata=datasets_metadata,
        processes=processes,
        data_processes=data_processes,
        systematics=systematics,
        year=args.year,
        category=cat,
    )
    datacard.dump(directory=args.output, card_name=f"datacard_{cat}_{args.year}.txt", shapes_filename=f"shapes_{cat}_{args.year}.root")
    datacards[f'datacard_{cat}_{args.year}.txt'] = datacard
    print(f"Datacard saved in {os.path.abspath(os.path.join(args.output, f'datacard_{cat}_{args.year}.txt'))}")

# Combine datacards
combine_datacards(datacards, directory=args.output)
