import os
import json
import argparse
from collections import defaultdict
from coffea.util import load, save
import hist

parser = argparse.ArgumentParser(description="Rescale nuisances")
parser.add_argument("-i", "--input", help="Coffea input file with histograms", required=True)
parser.add_argument("-o", "--output", help="Coffea output file with rescaled nuisances", default=None, required=False)
parser.add_argument("-v", "--verbose", help="Print debug information", action="store_true")
args = parser.parse_args()

if not args.output:
    args.output = args.input.replace(".coffea", "_renormalized.coffea")

samples_to_rescale = [
    'TTbbSemiLeptonic__TTbbSemiLeptonic_tt+B_4j_DCTR_L',
    'TTbbSemiLeptonic__TTbbSemiLeptonic_tt+B_4j_DCTR_M',
    'TTbbSemiLeptonic__TTbbSemiLeptonic_tt+B_4j_DCTR_H',
    'TTbbSemiLeptonic__TTbbSemiLeptonic_tt+B_5j_DCTR_L',
    'TTbbSemiLeptonic__TTbbSemiLeptonic_tt+B_5j_DCTR_M',
    'TTbbSemiLeptonic__TTbbSemiLeptonic_tt+B_5j_DCTR_H',
    'TTbbSemiLeptonic__TTbbSemiLeptonic_tt+B_6j_DCTR_L',
    'TTbbSemiLeptonic__TTbbSemiLeptonic_tt+B_6j_DCTR_M',
    'TTbbSemiLeptonic__TTbbSemiLeptonic_tt+B_6j_DCTR_H',
    'TTbbSemiLeptonic__TTbbSemiLeptonic_tt+B_>=7j_DCTR_L',
    'TTbbSemiLeptonic__TTbbSemiLeptonic_tt+B_>=7j_DCTR_M',
    'TTbbSemiLeptonic__TTbbSemiLeptonic_tt+B_>=7j_DCTR_H',
    'TTToSemiLeptonic__TTToSemiLeptonic_tt+LF',
    'TTToSemiLeptonic__TTToSemiLeptonic_tt+C'
]

nuisances = [
    'JER_AK4PFchs',
    'JES_Total_AK4PFchs',
    'pileup',
    'sf_jet_puId',
    'sf_ele_id',
    'sf_ele_reco',
    'sf_ele_trigger_era',
    'sf_ele_trigger_ht',
    'sf_ele_trigger_pileup',
    'sf_ele_trigger_stat',
    'sf_mu_id',
    'sf_mu_iso',
    'sf_mu_trigger',
    'sf_btag_withcalib_complete_ttsplit_cferr1',
    'sf_btag_withcalib_complete_ttsplit_cferr2',
    'sf_btag_withcalib_complete_ttsplit_hf',
    'sf_btag_withcalib_complete_ttsplit_hfstats1',
    'sf_btag_withcalib_complete_ttsplit_hfstats2',
    'sf_btag_withcalib_complete_ttsplit_lf',
    'sf_btag_withcalib_complete_ttsplit_lfstats1',
    'sf_btag_withcalib_complete_ttsplit_lfstats2',
    'sf_lhe_pdf_weight',
    'sf_partonshower_fsr',
    'sf_partonshower_isr',
    'sf_qcd_factor_scale',
    'sf_qcd_renorm_scale'
]

categories_analysis = ["CR", "CR_ttlf_0p60", "CR_ttcc", "SR"]
df = load(args.input)
years = list(df["datasets_metadata"]["by_datataking_period"].keys())

# We take a reference histogram to compute the rescaling factors
# In this case, we take the number of jets, but the choice is arbitrary
histo = df["variables"]["nJets"]

year = "2018"
rescaling_factor = defaultdict(dict)
for year in years:
    rescaling_factor[year] = defaultdict(dict)
    for nuisance in nuisances:
        for sample in samples_to_rescale:
            datasets = df["datasets_metadata"]["by_datataking_period"][year][sample]
            for dataset in datasets:
                h = histo[sample][dataset]
                axis_cat = h.axes[0]
                for systematic in nuisances:
                    for shift in ["Up", "Down"]:
                        variation = systematic + shift
                        # Rescale the varied shape such that the integral of the nominal shape is preserved
                        num = sum([sum(h[{'cat' : cat, 'variation' : "nominal"}].values(flow=True)) for cat in categories_analysis])
                        den = sum([sum(h[{'cat' : cat, 'variation' : variation}].values(flow=True)) for cat in categories_analysis])
                        rescaling_factor[year][sample][variation] = num / den

# Loop over all the histograms and rescale the nuisances by the rescaling factors
for histname, hist_to_rescale in df["variables"].items():
    for year, rescaling_factor_year in rescaling_factor.items():
        for sample, dict_rescaling in rescaling_factor_year.items():
            for dataset in df["datasets_metadata"]["by_datataking_period"][year][sample]:
                h = hist_to_rescale[sample][dataset]
                new_histogram = hist.Hist(*h.axes, storage=hist.storage.Weight())
                new_histogram_view = new_histogram.view(flow=True)
                for variation in h.axes["variation"]:
                    variation_index = new_histogram.axes["variation"].index(variation)
                    if variation in dict_rescaling.keys():
                        factor = dict_rescaling[variation]
                        if args.verbose:
                            print("Year: ", year, "Sample: ", sample, "Dataset: ", dataset, "Variation: ", variation, "Factor: ", factor)
                            print(f"{histname} (nominal) histogram CR+CR_ttlf+SR: \n", sum([h[{'cat' : c, 'variation' : "nominal"}] for c in categories_analysis]))
                            print(f"{histname} ({variation}) histogram CR+CR_ttlf+SR, before rescaling: \n", sum([h[{'cat' : c, 'variation' : variation}] for c in categories_analysis]))
                        for cat in h.axes["cat"]:
                            cat_index = new_histogram.axes["cat"].index(cat)
                            # Renormalize histograms in the analysis categories
                            if cat in categories_analysis:
                                new_histogram_view[cat_index, variation_index, :] = (h[cat, variation, :]*factor).view(flow=True)
                            else:
                                new_histogram_view[cat_index, variation_index, :] = h[cat, variation, :].view(flow=True)
                        if args.verbose:
                            print(f"Rescaled {histname} ({variation}) histogram CR+CR_ttlf+SR, after rescaling: \n", sum([new_histogram[{'cat' : c, 'variation' : variation}] for c in categories_analysis]))
                            print("")
                    else:
                        for cat in h.axes["cat"]:
                            cat_index = new_histogram.axes["cat"].index(cat)
                            new_histogram_view[cat_index, variation_index, :] = h[cat, variation, :].view(flow=True)
                df["variables"][histname][sample][dataset] = new_histogram

# Save the rescaled histograms in a new .coffea file
print(f"Saving rescaled histograms in {args.output}")
save(df, args.output)
basefolder = os.path.dirname(args.output)
filename_rescaling = os.path.join(basefolder, "rescaling_factors.json")
print(f"Saving rescaling factors in {filename_rescaling}")
with open(filename_rescaling, "w") as f:
    json.dump(rescaling_factor, f, indent=4)
