#!/bin/bash

commands=(
    "python -m ALIGN_FL.runner.sim_align_fl_vae -i align_fl_out_percent5_no_dp_lip0 -c 5 -dp no_dp -dim 2 -r 5 -lip 0 -dt mnist"
    "python -m ALIGN_FL.runner.sim_align_fl_vae -i align_fl_out_percent5_no_dp_lip0.1 -c 5 -dp no_dp -dim 2 -r 5 -lip 0.1 -dt mnist"
    "python -m ALIGN_FL.runner.sim_align_fl_vae -i align_fl_out_percent5_full_dp_lip0 -c 5 -dp full_dp -dim 2 -r 5 -lip 0 -dt mnist"
    "python -m ALIGN_FL.runner.sim_align_fl_vae -i align_fl_out_percent5_no_dp_lip0 -c 5 -dp no_dp -dim 2 -r 5 -lip 0 -dt fmnist"
    "python -m ALIGN_FL.runner.sim_align_fl_vae -i align_fl_out_percent5_no_dp_lip0.1 -c 5 -dp no_dp -dim 2 -r 5 -lip 0.1 -dt fmnist"
    "python -m ALIGN_FL.runner.sim_align_fl_vae -i align_fl_out_percent5_full_dp_lip0 -c 5 -dp full_dp -dim 2 -r 5 -lip 0 -dt fmnist"
    "python -m ALIGN_FL.runner.sim_fed_avg_vae -i fedavg_out_percent5_gp0 -c 5 -dp no_dp -dim 2 -r 5 -lip 0 -dt mnist"
    "python -m ALIGN_FL.runner.sim_fed_prox_vae -i fedprox_term1_out_percent5_gp0 -c 5 -dp no_dp -dim 2 -r 5 -lip 0 -dt mnist"

)

for cmd in "${commands[@]}"; do
    echo "Running: $cmd"
    eval $cmd
    if [ $? -ne 0 ]; then
        echo "Command failed: $cmd"
        exit 1
    fi
done