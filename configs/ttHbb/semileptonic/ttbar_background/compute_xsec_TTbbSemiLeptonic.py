# Script to compute the cross section of the 4FS MC sample from the 5FS sample
from coffea.util import load

import argparse

xsec_5FS = 365.4574 # pb

parser = argparse.ArgumentParser(description='Compute the cross section of the 4FS MC sample from the 5FS sample')

parser.add_argument('--input', '-i', type=str, help='Input .coffea file')
parser.add_argument('--year', type=str, help='Data taking year')

args = parser.parse_args()

accumulator = load(args.input)

w = accumulator['sum_genweights']

# Compute the ratio between the 5FS and 4FS cross sections from the sum of genweights
ratio = w[f'TTToSemiLeptonic_{args.year}'] / w[f'TTbbSemiLeptonic_4f_{args.year}']

print("*"*80)
print("Ratio from sum of genweights", end="\n\n")
print(f"TTToSemiLeptonic 5FS cross section: {xsec_5FS} pb")
print(f"TTbbSemiLeptonic 4FS cross section: {round(xsec_5FS / ratio, 4)} pb")
print(f"Ratio: {ratio}")

# Compute the ratio between the 5FS and 4FS cross sections in the SingleEle_1b category from the histogram of the number of leptons
h_5f = accumulator['variables']['nLepton']['TTToSemiLeptonic_2018'][{'cat' : 'SingleEle_1b'}]
h_4f = accumulator['variables']['nLepton']['TTbbSemiLeptonic_4f_2018'][{'cat' : 'SingleEle_1b'}]

ratio = sum(h_5f.values()[()]) / sum(h_4f.values()[()])

print("*"*80)
print("Ratio in SingleEle_1b category", end="\n\n")

print(f"TTToSemiLeptonic 5FS cross section: {xsec_5FS} pb")
print(f"TTbbSemiLeptonic 4FS cross section: {round(xsec_5FS / ratio, 4)} pb")
print(f"Ratio: {ratio}")

