import os
import re
import argparse
import uproot
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")
plt.rcParams["font.size"] = 10

def get_matrix(histo, params):
    A = []
    for i, x in enumerate(params):
        row = []
        for j, y in enumerate(params):
            row.append(histo[{'xaxis':x, 'yaxis':y}])
        A.append(row)
    return np.matrix(A)

def get_corr(histo, param):
    row = {}
    for j, y in enumerate(histo.axes[0]):
        row[y] = histo[{'xaxis':param, 'yaxis':y}]
    return row

def plot_covariance_matrix(cov_matrix, labels=None, title="Covariance", output_folder='.',  cmap='coolwarm'):
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Plot the covariance matrix with colormap
    cax = ax.matshow(cov_matrix, cmap=cmap, vmin=-1, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(cax)
    cbar.set_label('Correlation')
    
    # Write values on the matrix cells
    for (i, j), val in np.ndenumerate(cov_matrix):
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='black' if abs(val) < 0.5 else 'white')
    # Set labels if provided
    if labels:
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticklabels(labels)
    
    plt.title(f'{title} Matrix', fontsize=16)
    filename = os.path.join(output_folder, f"{title.lower()}_matrix.png")
    plt.savefig(filename, dpi=400)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot covariance matrix from dctr fit")
    parser.add_argument("filename", type=str, help="Path to the robustHesse file containing the covariance matrix")
    parser.add_argument("-o", "--output", type=str, help="Output directory for the plots", default=None, required=False)
    parser.add_argument("--params", type=str, nargs="+", help="List of parameters to plot", default=None, required=False)
    args = parser.parse_args()
    if not os.path.basename(args.filename).startswith("robustHesse"):
        raise ValueError("The input file must be the output of the robustHesse command")
    if not os.path.basename(args.filename).endswith(".root"):
        raise ValueError("The input file must be a ROOT file")
    if args.output is None:
        args.output = os.path.dirname(args.filename)
    f = uproot.open(args.filename)
    for matrix_name in ["Covariance", "Correlation"]:
        h = f[f"h_{matrix_name.lower()}"]
        histo = h.to_hist()
        if args.params:
            params = [p for p in histo.axes[0] if any([re.match(param, p) for param in args.params])]
        else:
            params = [p for p in histo.axes[0] if p.startswith('SF')] + ['r']
        cov = get_matrix(histo, params)
        plot_covariance_matrix(cov, params, title=matrix_name, output_folder=args.output)
    corr_r = get_corr(histo, 'r')
    dict(sorted(corr_r.items(), key=lambda item: item[1], reverse=True))