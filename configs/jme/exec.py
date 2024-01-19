import subprocess
import argparse
from params.binning import eta_bins
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
    help="Sign of eta bins (+, -, all)",
    type=str,
    default="-",
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
args = parser.parse_args()

args.flavsplit = int(args.flavsplit)
args.pnet = int(args.pnet)

# Define a list of eta bins
eta_bins = eta_bins if not args.inclusive_eta else None

test = "--test" if args.test else ""

if args.cartesian:
    print("Running cartesian multicuts")
    sign = args.sign

    dir_name = (
        f"out_cartesian_{'neg' if sign=='-' else ('pos' if sign=='+' else 'all')}{'_flavsplit' if args.flavsplit else f'_{args.flav}'}{'_pnet' if args.pnet else ''}{'_test' if args.test else ''}"
        if not args.dir
        else args.dir
    )
    tmux_session = (
        "cartesian_neg"
        if sign == "-"
        else ("cartesian_pos" if sign == "+" else "cartesian_all")
    )

    command0 = f"tmux kill-session -t {tmux_session}"
    command1 = f'tmux new-session -d -s {tmux_session} && tmux send-keys "export CARTESIAN=1 && export SIGN={args.sign} && export FLAVSPLIT={args.flavsplit} && export PNET={args.pnet} && export FLAV={args.flav}" "C-m"'
    command2 = f'tmux send-keys "pocket_coffea" "C-m" "time runner.py --cfg cartesian_config.py {test} --full -o {dir_name}" "C-m"'
    command3 = f'tmux send-keys "make_plots.py out_cartesian --overwrite -j 8" "C-m"'

    subprocess.run(command0, shell=True)
    print(f"killed session {tmux_session}")
    if not args.kill:
        subprocess.run(command1, shell=True)
        subprocess.run(command2, shell=True)
        # subprocess.run(command3, shell=True)
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
                command2 = f'tmux send-keys "pocket_coffea" "C-m" "time runner.py --cfg jme_config.py  {test} --full -o out_separate_eta_bin/eta{eta_bin_min}to{eta_bin_max}" "C-m"'
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
            command5 = f'tmux send-keys "pocket_coffea" "C-m"'
            subprocess.run(command5, shell=True)
            for i in range(len(eta_bins) - 1):
                eta_bin_min = eta_bins[i]
                eta_bin_max = eta_bins[i + 1]
                dir_name = (
                    f"out_separate_eta_bin_seq{'_pnet' if args.pnet else ''}{'_test' if args.test else ''}/eta{eta_bin_min}to{eta_bin_max}"
                    if not args.dir
                    else args.dir
                )
                command2 = f'tmux send-keys "export ETA_MIN={eta_bin_min} && export ETA_MAX={eta_bin_max} && export PNET={args.pnet}" "C-m" "echo $ETA_MIN" "C-m" "echo $ETA_MAX" "C-m"'
                command3 = f'tmux send-keys "time runner.py --cfg jme_config.py  {test} --full -o {dir_name}" "C-m"'
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
        command2 = f'tmux send-keys "pocket_coffea" "C-m" "time runner.py --cfg jme_config.py  {test} --full -o out_inclusive_eta" "C-m"'
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
