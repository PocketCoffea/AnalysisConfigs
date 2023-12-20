import subprocess
import argparse
from params.binning import eta_bins

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
args = parser.parse_args()

# Define a list of eta bins
eta_bins = eta_bins if not args.inclusive_eta else None

if args.cartesian:
    print("Running cartesian multicuts")
    command0 = f"tmux kill-session -t cartesian"
    command1 = f"tmux new-session -d -s cartesian"
    command2 = f'tmux send-keys "pocket_coffea" "C-m" "time runner.py --cfg cartesian_config.py --full -o out_cartesian -lf 10" "C-m"'
    command3 = f'tmux send-keys "make_plots.py out_cartesian --overwrite -j 8" "C-m"'

    subprocess.run(command0, shell=True)
    print("killed session cartesian")
    if not args.kill:
        subprocess.run(command1, shell=True)
        subprocess.run(command2, shell=True)
        # subprocess.run(command3, shell=True)
        print("tmux attach -t cartesian")

else:
    # Loop over the eta bins
    if eta_bins:
        print(f"Running over eta bins {eta_bins}")
        for i in range(len(eta_bins) - 1):
            eta_bin_min = eta_bins[i]
            eta_bin_max = eta_bins[i + 1]

            comand0 = f"tmux kill-session -t {eta_bin_min}to{eta_bin_max}"
            command1 = f'tmux new-session -d -s {eta_bin_min}to{eta_bin_max} && tmux send-keys "export ETA_MIN={eta_bin_min}" "C-m" "export ETA_MAX={eta_bin_max}" "C-m" "echo $ETA_MIN" "C-m" "echo $ETA_MAX" "C-m"'
            command2 = f'tmux send-keys "pocket_coffea" "C-m" "time runner.py --cfg jme_config.py --full -o out_separate_eta_bin/eta{eta_bin_min}to{eta_bin_max}" "C-m"'
            command3 = f'tmux send-keys "make_plots.py out_separate_eta_bin/eta{eta_bin_min}to{eta_bin_max} --overwrite -j 1" "C-m"'
            subprocess.run(comand0, shell=True)
            print(f"killed session {eta_bin_min}to{eta_bin_max}")
            if not args.kill:
                subprocess.run(command1, shell=True)
                subprocess.run(command2, shell=True)
                # subprocess.run(command3, shell=True)
                print(f"tmux attach -t {eta_bin_min}to{eta_bin_max}")

    else:
        print("No eta bins defined")
        print("Running over inclusive eta")
        comand0 = f"tmux kill-session -t inclusive_eta"
        command1 = f"tmux new-session -d -s inclusive_eta"
        command2 = f'tmux send-keys "pocket_coffea" "C-m" "time runner.py --cfg jme_config.py --full -o out_inclusive_eta -lf 1" "C-m"'
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
