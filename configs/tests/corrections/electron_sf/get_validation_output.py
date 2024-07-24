# Extract txt file with run,lumi,event,weight for the validation

from coffea.util import load
import argparse

parser = argparse.ArgumentParser(description='Get validation output')
parser.add_argument('--input', type=str, help='Input file')
parser.add_argument('--output', type=str, help='Output file')
parser.add_argument('--sample', type=str, help='Sample name')
parser.add_argument('--year', type=str, help='Year')
args = parser.parse_args()


df = load(args.input)

weight =df["columns"][args.sample][f"{args.sample}_{args.year}"]["inclusive"]["weight"].value
run = df["columns"][args.sample][f"{args.sample}_{args.year}"]["inclusive"]["events_run"].value
lumi = df["columns"][args.sample][f"{args.sample}_{args.year}"]["inclusive"]["events_luminosityBlock"].value
event = df["columns"][args.sample][f"{args.sample}_{args.year}"]["inclusive"]["events_event"].value

with open(args.output, "w") as f:
    f.write("run,lumi,event,weight\n")
    for i in range(len(weight)):
        f.write(f"{run[i]},{lumi[i]},{event[i]},{weight[i]}\n")

print(f"Output written to {args.output}")
