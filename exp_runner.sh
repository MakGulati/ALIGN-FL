#!/bin/bash

commands=(
    # "python -m ALIGN_FL.runner.sim_align_fl_vae -i align_fl_out_percent5_no_dp_lip0 -c 5 -dp no_dp -dim 2 -r 5 -lip 0 -dt mnist"
    # "python -m ALIGN_FL.runner.sim_align_fl_vae -i align_fl_out_percent5_no_dp_lip0.1 -c 5 -dp no_dp -dim 2 -r 5 -lip 0.1 -dt mnist"
    # "python -m ALIGN_FL.runner.sim_align_fl_vae -i align_fl_out_percent5_full_dp_lip0 -c 5 -dp full_dp -dim 2 -r 5 -lip 0 -dt mnist"
    # ALIGN_FL with new parameters
    # "python -m ALIGN_FL.runner.sim_align_fl_vae -i align_fl_kl1.0_synsamp2000_epochs10 -c 5 -dp no_dp -dim 2 -r 5 -lip 0 -dt mnist --lambda_kl 1.0 --synthetic_samples 5000 --server_epochs 10"
    # "python -m ALIGN_FL.runner.sim_align_fl_vae -i align_fl_kl0.1_synsamp1000_epochs5 -c 5 -dp no_dp -dim 2 -r 5 -lip 0 -dt mnist --lambda_kl 0.1 --synthetic_samples 1000 --server_epochs 5"
    # "python -m ALIGN_FL.runner.sim_align_fl_vae -i align_fl_out_percent5_no_dp_lip0 -c 5 -dp no_dp -dim 2 -r 5 -lip 0 -dt fmnist"
    # "python -m ALIGN_FL.runner.sim_align_fl_vae -i align_fl_out_percent5_no_dp_lip0.1 -c 5 -dp no_dp -dim 2 -r 5 -lip 0.1 -dt fmnist"
    # "python -m ALIGN_FL.runner.sim_align_fl_vae -i align_fl_out_percent5_full_dp_lip0 -c 5 -dp full_dp -dim 2 -r 5 -lip 0 -dt fmnist"
    # "python -m ALIGN_FL.runner.sim_fed_avg_vae -i fedavg_out_percent5_gp0 -c 5 -dp no_dp -dim 2 -r 5 -lip 0 -dt mnist"
    # "python -m ALIGN_FL.runner.sim_fed_prox_vae -i fedprox_term1_out_percent5_gp0 -c 5 -dp no_dp -dim 2 -r 5 -lip 0 -dt mnist"
    # "python -m ALIGN_FL.runner.sim_moon_vae -i moon_out_percent5_gp0 -c 5 -dp no_dp -dim 2 -r 20 -lip 0 -dt mnist"
    # "python -m ALIGN_FL.runner.sim_moon_vae -i moon_out_percent5_temp0.1 -c 5 -dp no_dp -dim 2 -r 20 -lip 0 --temperature 0.1 -dt mnist"
    # "python -m ALIGN_FL.runner.sim_moon_vae -i moon_out_percent5_mu2.0 -c 5 -dp no_dp -dim 2 -r 20 -lip 0 --mu 2.0 -dt mnist"
    # "python -m ALIGN_FL.runner.sim_moon_vae -i moon_out_percent5_full_dp -c 5 -dp full_dp -dim 2 -r 20 -lip 0 -dt mnist"
    
    # Hybrid MOON-ALIGN runs
    # "python -m ALIGN_FL.runner.sim_hybrid_moon_align -i hybrid_moon_align_baseline -c 5 -dp no_dp -dim 2 -r 20 -lip 0 -dt mnist --temperature 0.5 --mu 1.0 --buffer_size 5 --synthetic_samples 1000 --server_epochs 5 --lambda_kl 0.1"
    # "python -m ALIGN_FL.runner.sim_hybrid_moon_align -i hybrid_moon_align_kl1 -c 5 -dp no_dp -dim 2 -r 20 -lip 0 -dt mnist --temperature 0.5 --mu 1.0 --buffer_size 5 --synthetic_samples 5000 --server_epochs 5 --lambda_kl 1"
    # Non-averaging mode (stateful server like ALIGN_FL)
    "python -m ALIGN_FL.runner.sim_hybrid_moon_align -i hybrid_moon_align_stateful_kl1 -c 5 -dp no_dp -dim 2 -r 20 -lip 0 -dt mnist --temperature 0.5 --mu 1.0 --buffer_size 5 --synthetic_samples 5000 --server_epochs 5 --lambda_kl 1 --no_averaging"
    # "python -m ALIGN_FL.runner.sim_hybrid_moon_align -i hybrid_moon_align_temp0.1 -c 5 -dp no_dp -dim 2 -r 20 -lip 0 -dt mnist --temperature 0.1 --mu 1.0 --buffer_size 5 --synthetic_samples 1000 --server_epochs 5 --lambda_kl 0.1"
    # "python -m ALIGN_FL.runner.sim_hybrid_moon_align -i hybrid_moon_align_mu2.0 -c 5 -dp no_dp -dim 2 -r 20 -lip 0 -dt mnist --temperature 0.5 --mu 2.0 --buffer_size 5 --synthetic_samples 1000 --server_epochs 5 --lambda_kl 0.1"
    # "python -m ALIGN_FL.runner.sim_hybrid_moon_align -i hybrid_moon_align_samples2000 -c 5 -dp no_dp -dim 2 -r 20 -lip 0 -dt mnist --temperature 0.5 --mu 1.0 --buffer_size 5 --synthetic_samples 2000 --server_epochs 5 --lambda_kl 0.1"
    # "python -m ALIGN_FL.runner.sim_hybrid_moon_align -i hybrid_moon_align_server_epochs10_kl1 -c 5 -dp no_dp -dim 2 -r 20 -lip 0 -dt mnist --temperature 0.5 --mu 1.0 --buffer_size 5 --synthetic_samples 1000 --server_epochs 10 --lambda_kl 1"
    # "python -m ALIGN_FL.runner.sim_hybrid_moon_align -i hybrid_moon_align_stateful_full_dp -c 5 -dp full_dp -dim 2 -r 20 -lip 0 -dt mnist --temperature 0.5 --mu 1.0 --buffer_size 5 --synthetic_samples 1000 --server_epochs 5 --lambda_kl 1 --no_averaging"
)

for cmd in "${commands[@]}"; do
    echo "Running: $cmd"
    eval $cmd
    if [ $? -ne 0 ]; then
        echo "Command failed: $cmd"
        exit 1
    fi
done