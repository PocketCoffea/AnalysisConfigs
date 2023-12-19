#!/bin/bash

# Define a list of eta bins
eta_bins="-5.191,-4.889,-4.716,-4.538,-4.363,-4.191,-4.013,-3.839,-3.664,-3.489,-3.314,-3.139,-2.964,-2.853,-2.65,-2.5,-2.322,-2.172,-2.043,-1.93,-1.83,-1.74,-1.653,-1.566,-1.479,-1.392,-1.305,-1.218,-1.131,-1.044,-0.957,-0.879,-0.783,-0.696,-0.609,-0.522,-0.435,-0.348,-0.261,-0.174,-0.087,0.0,0.087,0.174,0.261,0.348,0.435,0.522,0.609,0.696,0.783,0.879,0.957,1.044,1.131,1.218,1.305,1.392,1.479,1.566,1.653,1.74,1.83,1.93,2.043,2.172,2.322,2.5,2.65,2.853,2.964,3.139,3.314,3.489,3.664,3.839,4.013,4.191,4.363,4.538,4.716,4.889,5.191"

# Loop over the eta bins

for i in len(eta_bins):
    eta_bin_min = eta_bins[i]
    eta_bin_max = eta_bins[i+1]
    echo "Running over eta bin $eta_bin_min to $eta_bin_max"
    # Create a new tmux window and run the command with the eta bin as an argument
    # tmux new-window "time runner.py --cfg jme_config.py --full -o out_single_eta_bin -lf 1 -lc 1 -e futures --eta-bin $eta_bin_min,$eta_bin_max"
done
