import os
import matplotlib.pyplot as plt
import mplhep as hep



# def plot_info(type, ax, fig, flav, variable, eta_sign, eta_bin, pt_bin=None, ax_ratio=None, ax_tot=None, fig_tot=None):
#     # write axis name in latex
#     ax.set_xlabel(f"Response" if type == "histo" else r"$p_{T}^{Gen}$ [GeV]")
#     ax.set_ylabel(f"Events" if type == "histo" else (
#         f"Median (Response)"

#     ))
#     # remove border of legend
#     ax.legend(frameon=False, ncol=2)

#     plt.grid(color="gray", linestyle="--", linewidth=0.5, which="both")
#     # hep.style.use("CMS")
#     hep.cms.label(
#         year="2022",
#         com="13.6",
#         label=f"Private Work",
#     )
#     # write a string on the plot
#     ax.text(
#         0.95,
#         0.7,
#         f"{correct_eta_bins[eta_bin]} <"
#         + r"$\eta^{Gen}$"
#         + f"< {correct_eta_bins[eta_bin+1]}\n"
#             + f" {int(pt_bins[pt_bin])} <"
#             + r"$p_{T}^{Gen}$"
#             + f"< {int(pt_bins[pt_bin+1])}",
#         horizontalalignment="right",
#         verticalalignment="top",
#         transform=ax.transAxes,
#     )


#     if args.full:
#         response_dir = (
#             f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/response_plots_unbinned"
#             if args.unbinned
#             else f"{main_dir}/{eta_sign}eta_{flav}flav_pnet/response_plots_binned"
#         )
#         os.makedirs(f"{response_dir}", exist_ok=True)

#     fig.savefig(
#         f"{response_dir}/histos_{variable}_{flav}_eta{correct_eta_bins[eta_bin]}to{correct_eta_bins[eta_bin+1]}_pt{pt_bins[pt_bin]}to{pt_bins[pt_bin+1]}.png",
#         bbox_inches="tight",
#         dpi=300,
#     )
#     plt.close(fig)
