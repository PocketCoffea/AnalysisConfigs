from coffea.util import load

import argparse

parser = argparse.ArgumentParser(description="Run the jme analysis")
parser.add_argument(
    "-i",
    "--input",
    type=str,
    help="Input dir",
)
args = parser.parse_args()

o = load(f"{args.input}/output_all.coffea")

print(o["variables"].keys())
h=o["variables"]["MatchedJets_ResponseVSpt"]["QCD"]["QCD_PT-15to7000_PT-15to7000_2018"]
print(h)
print("\n")
print(o["columns"].keys())
# print first value of dict
c=o["columns"]["QCD"]["QCD_PT-15to7000_PT-15to7000_2018"]
print(c.keys())
b=c[list(c.keys())[1]]
a=list(c[list(c.keys())[1]].keys())
print(a)
d=b[a[6]].value
print(a[6])
print(d)
print(d[d != -999.])
