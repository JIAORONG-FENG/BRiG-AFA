# BRiG-AFA

Bellman Risk-to-Go Learning for Non-Myopic Active Feature Acquisition.

BRiG-AFA learns budget-specific candidate risk-to-go functions by fitted Bellman regression. At inference, the policy uses only the observed feature values, observation mask, candidate identity, and remaining budget. It does not use labels, unobserved candidate values, or task-specific metadata.

## Repository layout

```text
briga/              Core models, training, evaluation, and data loaders
paper/              Overleaf-compatible LaTeX source and compiled manuscript
```

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Reproducing the experiments

Run commands from the repository root.

### CUBE-NM

```bash
python scripts/run_cube_nm_final_multiseed.py \
  --seeds 1 3 5 7 9 \
  --budgets 1 2 3 5 8
```

### Fashion-MNIST with 20 candidate pixels

```bash
python scripts/run_fashion_mnist_selected_probe.py \
  --seeds 1 3 5 7 9 \
  --budgets 1 2 4 8 12 16 20
```

Fashion-MNIST is downloaded automatically into `BRIGA_DATA_CACHE` when set, or `~/.cache/briga_afa/torchvision` otherwise.

### MiniBooNE

```bash
python scripts/run_miniboone_tabular.py \
  --seeds 1 3 5 \
  --budgets 1 2 4 8 16 \
  --max-samples 30000
```

MiniBooNE is downloaded from the UCI Machine Learning Repository. Set `BRIGA_DATA_CACHE` to override the default `~/.cache/briga_afa` location.

