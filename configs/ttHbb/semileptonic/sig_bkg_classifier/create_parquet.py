import os
import argparse
import awkward as ak

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, required=True)
args = parser.parse_args()

#basedir = "/work/mmarcheg/ttHbb/EFT/sig_bkg_classifier/sig_bkg_ntuples_ttHTobb_ttToSemiLep_improved_matching/ntuples/output_columns_parton_matching/parton_matching_20_06_24"
datasets =os.listdir(args.input)
for d in datasets:
    ak.to_parquet.dataset(os.path.join(args.input, d, "semilep"))
