#!/bin/bash

# This script runs ALIGN-FL experiments with varying privacy parameters
# to generate data for privacy-utility tradeoff plots

# Create output directory for results
RESULTS_DIR="privacy_utility_results"
mkdir -p $RESULTS_DIR

# Set common parameters
NUM_CLIENTS=5
ROUNDS=10
LATENT_DIM=2
DATASET="mnist"
SYNTH_SAMPLES=5000
SERVER_EPOCHS=10
LAMBDA_KL=1

echo "Running ALIGN-FL experiments with varying Lipschitz constants (λ)..."

# Experiment set 1: Varying Lipschitz constants (λ)
# λ values: 0, 0.01, 0.05, 0.1, 0.5, 1.0
LIP_VALUES=(0 0.01 0.05 0.1 0.5 1.0)

for lip in "${LIP_VALUES[@]}"; do
    IDENTIFIER="align_fl_lip${lip}_tradeoff"
    echo "Running experiment with λ=$lip"
    
    cmd="python -m ALIGN_FL.runner.sim_align_fl_vae -i $IDENTIFIER -c $NUM_CLIENTS -dp no_dp -dim $LATENT_DIM -r $ROUNDS -lip $lip -dt $DATASET --synthetic_samples $SYNTH_SAMPLES --server_epochs $SERVER_EPOCHS --lambda_kl $LAMBDA_KL"
    
    echo "Executing: $cmd"
    eval $cmd
    
    if [ $? -ne 0 ]; then
        echo "Command failed: $cmd"
        echo "Continuing with next experiment..."
    fi
done

echo "Running ALIGN-FL experiments with varying privacy budgets (ε)..."

# Experiment set 2: Varying epsilon values in full_dp mode
# ε values: 0, 1, 5, 10, 15, 20
EPSILON_VALUES=(0 1 5 10 15 20)

for eps in "${EPSILON_VALUES[@]}"; do
    IDENTIFIER="align_fl_eps${eps}_tradeoff"
    echo "Running experiment with ε=$eps"
    
    cmd="python -m ALIGN_FL.runner.sim_align_fl_vae -i $IDENTIFIER -c $NUM_CLIENTS -dp full_dp -dim $LATENT_DIM -r $ROUNDS -lip 0 -dt $DATASET --synthetic_samples $SYNTH_SAMPLES --server_epochs $SERVER_EPOCHS --lambda_kl $LAMBDA_KL --epsilon $eps"
    
    echo "Executing: $cmd"
    eval $cmd
    
    if [ $? -ne 0 ]; then
        echo "Command failed: $cmd"
        echo "Continuing with next experiment..."
    fi
done

echo "All experiments completed."
echo "Creating privacy-utility tradeoff plots..."

# Run the Python script to generate plots
python -m ALIGN_FL.utils.privacy_utility_plots --results_dir=$RESULTS_DIR

echo "Done!"
