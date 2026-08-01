# BRiG-AFA

Bellman Risk-to-Go Learning for Non-Myopic Active Feature Acquisition.

BRiG-AFA learns budget-specific candidate risk-to-go functions by fitted Bellman regression. At inference, the policy uses only the observed feature values, observation mask, candidate identity, and remaining budget. It does not use labels, unobserved candidate values, or task-specific metadata.

## Repository layout

```text
briga/              Core models, training, evaluation, and data loaders
scripts/            Reproduction and figure-generation entry points
results/*.csv       Seed-level and summarized results used by the paper
results/tables/     Compact publication tables
results/figures/    Publication figures in PDF and PNG formats
paper/              Overleaf-compatible LaTeX source and compiled manuscript
```

Model checkpoints, downloaded datasets, caches, training logs, and large diagnostic trajectory files are intentionally excluded.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

PyTorch installation can be adjusted for the local CUDA version by following the official PyTorch installation instructions.

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

### Regenerating publication figures

```bash
python scripts/make_publication_figures_route_a.py
```

The script reads the committed compact CSV files and writes figures to `results/publication_route_a/`. The committed final figures are also available under `results/figures/`.

## Main empirical result

On Fashion-MNIST with 20 candidate pixels, BRiG-AFA improves test accuracy over the matched one-step Myopic-Q ablation by `10.20 ± 0.74` percentage points at budget 4 over five paired seeds. See the manuscript for the complete experimental scope and limitations.

## Paper

The Overleaf-compatible source is in [`paper/main.tex`](paper/main.tex), with a compiled copy at [`paper/main.pdf`](paper/main.pdf).

## Citation

Citation metadata is provided in [`CITATION.cff`](CITATION.cff).

## License

No open-source license has been selected yet. Add an appropriate license before distributing or accepting external contributions.
