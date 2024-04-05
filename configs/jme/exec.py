# python exec.py --full -pnet

import subprocess
import argparse
from params.binning import eta_bins, eta_sign_dict
import os
from time import sleep

parser = argparse.ArgumentParser(description="Run the jme analysis")
parser.add_argument(
    "--inclusive-eta",
    "-i",
    action="store_true",
    help="Run over eta bins",
    default=False,
)
parser.add_argument(
    "--kill", "-k", action="store_true", help="Kill all tmux sessions", default=False
)
parser.add_argument(
    "-c",
    "--cartesian",
    action="store_true",
    help="Run cartesian multicuts",
    default=False,
)
parser.add_argument(
    "-p", "--parallel", action="store_true", help="Run parallel eta bins", default=False
)
parser.add_argument(
    "-s",
    "--sign",
    help="Sign of eta bins" ,
    type=str,
    default="neg3",
)
parser.add_argument(
    "-fs",
    "--flavsplit",
    action="store_true",
    help="Flavour split",
    default=False,
)
parser.add_argument(
    "-pnet",
    "--pnet",
    action="store_true",
    help="Use ParticleNet regression",
    default=False,
)
parser.add_argument(
    "-t",
    "--test",
    action="store_true",
    help="Test run",
    default=False,
)
parser.add_argument(
    "-d",
    "--dir",
    help="Output directory",
    type=str,
    default="",
)
parser.add_argument(
    "-f",
    "--flav",
    help="Flavour",
    type=str,
    default="inclusive",
)
parser.add_argument(
    "--full",
    action="store_true",
    help="Run full cartesian analysis in all eta bins and all flavours sequentially",
    default=False,
)
parser.add_argument(
    "--plot",
    action="store_true",
    help="Make plots",
    default=False,
)
parser.add_argument(
    "--central",
    action="store_true",
    help="Central eta bin (-1.3, 1.3)",
    default=False,
)
args = parser.parse_args()

args.flavsplit = int(args.flavsplit)
args.pnet = int(args.pnet)
args.central = int(args.central)

# Define a list of eta bins
eta_bins = eta_bins if not args.inclusive_eta else None

executor = "--test" if args.test else "-e dask@T3_CH_PSI --custom-run-options params/t3_run_options.yaml"

eta_sign_list=list(eta_sign_dict.keys())
dir_prefix = os.environ.get("WORK", "") + "/out_jme/"
print("dir_prefix", dir_prefix)

def run_command(sign, flav, dir_name):
    command2 = f'tmux send-keys "export CARTESIAN=1 && export SIGN={sign} && export FLAVSPLIT={args.flavsplit} && export PNET={args.pnet} && export FLAV={flav} && export CENTRAL={args.central}" "C-m"'
    command3 = f'tmux send-keys "time pocket-coffea run --cfg cartesian_config.py {executor} --full -o {dir_name}" "C-m"'
    command4 = f'tmux send-keys "make_plots.py {dir_name} --overwrite -j 64" "C-m"'

    subprocess.run(command2, shell=True)
    subprocess.run(command3, shell=True)
    if args.plot:
        subprocess.run(command4, shell=True)

if args.cartesian or args.full:
    print(
        f"Running cartesian multicuts {'in full configuration sequentially' if args.full else ''}"
    )
    sign = args.sign
    flav= args.flav

    flavs_list = ["inclusive", "b", "c", "g", "uds"] if (args.full and args.central) else ["inclusive"]

    tmux_session = (
        "full_cartesian"
        if args.full
        else f"{sign}_cartesian"
    )
    command0 = f"tmux kill-session -t {tmux_session}"
    subprocess.run(command0, shell=True)
    print(f"killed session {tmux_session}")
    if not args.kill:
        command1 = f'tmux new-session -d -s {tmux_session} && tmux send-keys "micromamba activate pocket-coffea" "C-m"'
        subprocess.run(command1, shell=True)

    if args.full:
        for sign in eta_sign_list if not args.central else [""]:
            if sign == "all":
                continue
            for flav in flavs_list:
                dir_name = (
                    f"{dir_prefix}out_cartesian_full{'_test' if args.test else ''}{args.dir}/{sign if not args.central else 'central'}eta_{flav}flav{'_pnet' if args.pnet else ''}"
                )
                if not os.path.isfile(f"{dir_name}/output_all.coffea"):
                    print(f"{dir_name}")
                    run_command(sign, flav, dir_name)
    else:
        dir_name = (
            f"{dir_prefix}out_cartesian_{sign if not args.central else 'central'}eta{'_flavsplit' if args.flavsplit else f'_{args.flav}flav'}{'_pnet' if args.pnet else ''}{'_test' if args.test else ''}"
            if not args.dir
            else args.dir
        )
        if not os.path.isfile(f"{dir_name}/output_all.coffea"):
            print(f"{dir_name}")
            run_command(sign, flav, dir_name)


    print(f"tmux attach -t {tmux_session}")

else:
    # Loop over the eta bins
    if eta_bins:
        if args.parallel:
            print(f"Running over eta bins {eta_bins} in parallel")
            for i in range(len(eta_bins) - 1):
                eta_bin_min = eta_bins[i]
                eta_bin_max = eta_bins[i + 1]

                comand0 = f"tmux kill-session -t {eta_bin_min}to{eta_bin_max}"
                command1 = f'tmux new-session -d -s {eta_bin_min}to{eta_bin_max} && tmux send-keys "export ETA_MIN={eta_bin_min}" "C-m" "export ETA_MAX={eta_bin_max}" "C-m" "echo $ETA_MIN" "C-m" "echo $ETA_MAX" "C-m"'
                command2 = f'tmux send-keys "micromamba activate pocket-coffea" "C-m" "time pocket-coffea run --cfg jme_config.py  {executor} --full -o out_separate_eta_bin/eta{eta_bin_min}to{eta_bin_max}" "C-m"'
                command3 = f'tmux send-keys "make_plots.py out_separate_eta_bin/eta{eta_bin_min}to{eta_bin_max} --overwrite -j 1" "C-m"'
                subprocess.run(comand0, shell=True)
                print(f"killed session {eta_bin_min}to{eta_bin_max}")
                if not args.kill:
                    subprocess.run(command1, shell=True)
                    subprocess.run(command2, shell=True)
                    # subprocess.run(command3, shell=True)
                    print(f"tmux attach -t {eta_bin_min}to{eta_bin_max}")
        else:
            print(f"Running over eta bins {eta_bins} in sequence")
            comand0 = f"tmux kill-session -t eta_bins"
            command1 = f"tmux new-session -d -s eta_bins"
            # execute the commands
            subprocess.run(comand0, shell=True)
            subprocess.run(command1, shell=True)

            # os.system(comand0)
            # os.system(command1)
            print(f"tmux attach -t eta_bins")
            command5 = f'tmux send-keys "micromamba activate pocket-coffea" "C-m"'
            subprocess.run(command5, shell=True)
            for i in range(len(eta_bins) - 1):
                eta_bin_min = eta_bins[i]
                eta_bin_max = eta_bins[i + 1]
                dir_name = (
                    f"{dir_prefix}out_separate_eta_bin_seq{'_pnet' if args.pnet else ''}{'_test' if args.test else ''}/eta{eta_bin_min}to{eta_bin_max}"
                    if not args.dir
                    else args.dir
                )
                command2 = f'tmux send-keys "export ETA_MIN={eta_bin_min} && export ETA_MAX={eta_bin_max} && export PNET={args.pnet}" "C-m" "echo $ETA_MIN" "C-m" "echo $ETA_MAX" "C-m"'
                command3 = f'tmux send-keys "time pocket-coffea run --cfg jme_config.py  {executor} --full -o {dir_name}" "C-m"'
                # command4 = f'tmux send-keys "make_plots.py {dir_name} --overwrite -j 8" "C-m"'

                # os.environ["ETA_MIN"] = f"{eta_bin_min}"
                # os.environ["ETA_MAX"] = f"{eta_bin_max}"

                if not os.path.isfile(f"{dir_name}/output_all.coffea"):
                    print(f"{dir_name}")
                    subprocess.run(command2, shell=True)
                    subprocess.run(command3, shell=True)
                    # subprocess.run(comand4, shell=True)
                    # os.system(command3)

    else:
        print("No eta bins defined")
        print("Running over inclusive eta")
        comand0 = f"tmux kill-session -t inclusive_eta"
        command1 = f"tmux new-session -d -s inclusive_eta"
        command2 = f'tmux send-keys "micromamba activate pocket-coffea" "C-m" "time pocket-coffea run --cfg jme_config.py  {executor} --full -o out_inclusive_eta" "C-m"'
        command3 = (
            f'tmux send-keys "make_plots.py out_inclusive_eta --overwrite -j 8" "C-m"'
        )
        subprocess.run(comand0, shell=True)
        print("killed session inclusive_eta")
        if not args.kill:
            subprocess.run(command1, shell=True)
            subprocess.run(command2, shell=True)
            subprocess.run(command3, shell=True)
            print("tmux attach -t inclusive_eta")
