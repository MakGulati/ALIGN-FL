# ALIGN_FL

A Federated Learning framework implementing various strategies including ALIGN_FL, FedAvg, and FedProx, with support for differential privacy and various datasets.

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

### Command Line Interface

The framework provides several runner scripts for different federated learning strategies:

1. ALIGN_FL VAE:
```bash
python -m ALIGN_FL.runner.sim_align_fl_vae [options]
```

2. FedAvg VAE:
```bash
python -m ALIGN_FL.runner.sim_fed_avg_vae [options]
```

3. FedProx VAE:
```bash
python -m ALIGN_FL.runner.sim_fed_prox_vae [options]
```

### Common Options

- `-i`: Output directory name
- `-c`: Number of clients (e.g., 5)
- `-dp`: Differential privacy mode ('no_dp' or 'full_dp' or 'dec_dp')
- `-dim`: Latent Dimension size (e.g., 2)
- `-r`: Round number (e.g., 5)
- `-lip`: Lipschitz constant (e.g., 0, 0.1)
- `-dt`: Dataset type ('mnist' or 'fmnist')

### Example Commands

#### ALIGN_FL with MNIST
```bash
# No DP-SGD, Lip=0
python -m ALIGN_FL.runner.sim_align_fl_vae -i align_fl_out_percent5_no_dp_lip0 -c 5 -dp no_dp -dim 2 -r 5 -lip 0 -dt mnist

# No DP-SGD, Lip=0.1
python -m ALIGN_FL.runner.sim_align_fl_vae -i align_fl_out_percent5_no_dp_lip0.1 -c 5 -dp no_dp -dim 2 -r 5 -lip 0.1 -dt mnist

# Full DP-SGD, Lip=0
python -m ALIGN_FL.runner.sim_align_fl_vae -i align_fl_out_percent5_full_dp_lip0 -c 5 -dp full_dp -dim 2 -r 5 -lip 0 -dt mnist
```

#### ALIGN_FL with Fashion MNIST
```bash
python -m ALIGN_FL.runner.sim_align_fl_vae -i align_fl_out_percent5_no_dp_lip0 -c 5 -dp no_dp -dim 2 -r 5 -lip 0 -dt fmnist
```

#### Other Strategies
```bash
# FedAvg
python -m ALIGN_FL.runner.sim_fed_avg_vae -i fedavg_out_percent5_gp0 -c 5 -dp no_dp -dim 2 -r 5 -lip 0 -dt mnist

# FedProx
python -m ALIGN_FL.runner.sim_fed_prox_vae -i fedprox_term1_out_percent5_gp0 -c 5 -dp no_dp -dim 2 -r 5 -lip 0 -dt mnist
```

## Project Structure

```
ALIGN_FL/
├── src/ALIGN_FL/
│   ├── utils/          # Utility functions and helper classes
│   ├── strategy/       # Implementation of federated learning strategies
│   └── runner/         # Command-line runner scripts
├── tests/              # Unit tests
└── examples/           # Example usage and notebooks
```

## Features

- Multiple federated learning strategies:
  - ALIGN_FL with VAE
  - FedAvg with VAE
  - FedProx with VAE
- Differential privacy support
- Multiple dataset support:
  - MNIST
  - Fashion MNIST
- Configurable parameters:
  - Number of clients
  - Dimensions
  - Rounds
  - Lipschitz constants

## License

MIT License