import os
import argparse
import awkward as ak

from multiprocessing import Pool
from functools import partial

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, required=True, help="Input folder path")
parser.add_argument("-j", "--workers", type=int, default=8, help="Number of parallel workers")
parser.add_argument("-c", "--cat", type=str, default="semilep_LHE", help="Category")
parser.add_argument("-v", "--var", type=str, default="nominal", help="Variation")
parser.add_argument("--overwrite", action="store_true", help="Overwrite existing metadata")
args = parser.parse_args()

error_filename = "error_parquet_metadata.log"

def create_parquet_metadata(dataset, category, variation):
    # The datasets with subsamples have subfolders
    if dataset.startswith("TTToSemiLeptonic") or dataset.startswith("TTbbSemiLeptonic") or dataset.startswith("DATA"):
        subfolders = os.listdir(os.path.join(args.input, dataset))
        for subfolder in subfolders:
            print(f"Processing {dataset}/{subfolder}")
            dataset_path = os.path.join(args.input, dataset, subfolder, category, variation)
            if not os.path.exists(dataset_path):
                raise ValueError(f"The path {dataset_path} does not exist.")
            if os.path.exists(os.path.join(dataset_path, "_metadata")) and not args.overwrite:
                print(f"Metadata already exists for {dataset}/{subfolder}")
                return
            try:
                ak.to_parquet.dataset(dataset_path)
            except:
                print(f"Error processing {dataset}/{subfolder}")
                with open(error_filename, "a") as f:
                    f.write(f"{dataset}/{subfolder}\n")
    else:
        print(f"Processing {dataset}")
        dataset_path = os.path.join(args.input, dataset, category, variation)
        if os.path.exists(os.path.join(dataset_path, "_metadata")) and not args.overwrite:
            print(f"Metadata already exists for {dataset}")
            return
        try:
            ak.to_parquet.dataset(dataset_path)
        except:
            print(f"Error processing {dataset}")
            with open(error_filename, "a") as f:
                f.write(f"{dataset}\n")

datasets =os.listdir(args.input)
# Parallelize the code: one process per dataset
if args.workers == 1:
    for dataset in datasets:
        create_parquet_metadata(dataset, args.cat, args.var)
else:
    with Pool(args.workers) as pool:
        pool.map(partial(create_parquet_metadata, category=args.cat, variation=args.var), datasets)