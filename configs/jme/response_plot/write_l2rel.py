import json
import sys

sys.path.append("../")
from params.binning import *

def write_l2rel_txt(main_dir, correct_eta_bins):
    # write a pol of 20th order in the ROOT format
    pol_string = "[0]+[1]*log10(x)+[2]*pow(log10(x),2)+[3]*pow(log10(x),3)+[4]*pow(log10(x),4)+[5]*pow(log10(x),5)+[6]*pow(log10(x),6)+[7]*pow(log10(x),7)+[8]*pow(log10(x),8)+[9]*pow(log10(x),9)+[10]*pow(log10(x),10)+[11]*pow(log10(x),11)+[12]*pow(log10(x),12)+[13]*pow(log10(x),13)+[14]*pow(log10(x),14)+[15]*pow(log10(x),15)+[16]*pow(log10(x),16)+[17]*pow(log10(x),17)+[18]*pow(log10(x),18)+[19]*pow(log10(x),19)+[20]*pow(log10(x),20) "

    # create txt
    flav='inclusive'
    with open(f"{main_dir}/l2rel.txt", "w") as l2_file:
        l2_file.write(f"{{1 JetEta 1 JetPt ({pol_string})  Correction L2Relative }}\n")
        for i in range(len(correct_eta_bins) - 1):
            try:
                with open(
                    f"{main_dir}/inv_median_plots_binned/fit_results_inverse_median_Response_inclusive_eta{correct_eta_bins[i]}to{correct_eta_bins[i+1]}.json",
                    "r",
                ) as f:
                    fit_results_dict = json.load(f)

                    params_string = ""
                    for param in fit_results_dict["inclusive_ResponsePNetRegNeutrino"]["parameters"]:
                        params_string += "    {:.8f}".format(param)
                    for j in range(21 - len(fit_results_dict["inclusive_ResponsePNetRegNeutrino"]["parameters"])):
                        params_string += " 0"
                    jetpt_low = "    {:.8f}".format(fit_results_dict["inclusive_ResponsePNetRegNeutrino"]["jet_pt"][0])
                    jetpt_up = "    {:.8f}".format(fit_results_dict["inclusive_ResponsePNetRegNeutrino"]["jet_pt"][1])
                    l2_file.write(
                        f" {correct_eta_bins[i]} {correct_eta_bins[i+1]} 23  {jetpt_low} {jetpt_up} {params_string}\n"
                    )
            except FileNotFoundError:
                print(f"File not found for {correct_eta_bins[i]} to {correct_eta_bins[i+1]}")
                #set parameter 0 to 1 and the rest to 0
                params_string = ''
                for j in range(21):
                    if j == 0:
                        params_string += ' 1'
                    else:
                        params_string += ' 0'
                l2_file.write(
                    f" {correct_eta_bins[i]} {correct_eta_bins[i+1]}    23  0 0 {params_string}\n"
                )
            except KeyError:
                print(f"No fit for {correct_eta_bins[i]} to {correct_eta_bins[i+1]}")
                #set parameter 0 to 1 and the rest to 0
                params_string = ''
                for j in range(21):
                    if j == 0:
                        params_string += '    1'
                    else:
                        params_string += '    0'
                l2_file.write(
                    f" {correct_eta_bins[i]} {correct_eta_bins[i+1]}    23  0 0 {params_string}\n"
                )


if __name__ == "__main__":
    write_l2rel_txt("/work/mmalucch/out_jme/out_cartesian_full_correctNeutrinosSeparation_jetpt_ZerosPtResponse/", eta_bins)