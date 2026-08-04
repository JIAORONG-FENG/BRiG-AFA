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

MIT License

Copyright (c) 2026 Jiaorong Feng

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
