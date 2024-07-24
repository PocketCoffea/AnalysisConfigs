import os
import awkward as ak

basedir = "/work/mmarcheg/ttHbb/EFT/sig_bkg_classifier/sig_bkg_ntuples_ttHTobb_ttToSemiLep_improved_matching/ntuples/output_columns_parton_matching/parton_matching_20_06_24"
datasets =os.listdir(basedir)
for d in datasets:
    ak.to_parquet.dataset(os.path.join(basedir, d, "semilep_LHE"))
