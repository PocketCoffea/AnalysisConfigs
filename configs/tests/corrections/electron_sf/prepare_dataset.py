import json
import argparse

parser = argparse.ArgumentParser(description='Setup dataset')
# sample
parser.add_argument('--sample', type=str, help='Sample name')
# year
parser.add_argument('--year', type=str, help='Year')
# file
parser.add_argument('--file', type=str, help='File')
# output
parser.add_argument('--output', type=str, help='Output file')
args = parser.parse_args()


dataset = {
    f"{args.sample}_{args.year}": {
	"metadata": {
	    "sample": args.sample,
	    "year": args.year,
	    "isMC": "True",
	    "nevents": 10000,
	    "xsec": 1.0
	},
	"files": [
	    args.file
	]
    }
}

with open(args.output, "w") as f:
    json.dump(dataset, f)
    
