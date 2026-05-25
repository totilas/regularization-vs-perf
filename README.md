# Optimal Regularization for Performative Learning

This repository contains code for the experiments in:

**Optimal Regularization for Performative Learning**  
Edwige Cyffers, Alireza Mirrokni, Marco Mondelli  
International Conference on Machine Learning (ICML), 2026

The main repository contains the original experiments for the paper. The folder
`neural_network_credit/` contains the additional neural-network experiment on the
`GiveMeSomeCredit` strategic-classification environment used for the rebuttal-stage
figure `metric_vs_lambda_final_acc.pdf`.

## Citation

```bibtex
@inproceedings{cyffers2026optimal,
  title     = {Optimal Regularization for Performative Learning},
  author    = {Cyffers, Edwige and Mirrokni, Alireza and Mondelli, Marco},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year      = {2026}
}
```

## Repository structure

```text
.
├── proportional/                 # Synthetic Proportional setting
├── real/                         # Real-data experiments
├── out/                          # Generated outputs, not tracked
├── neural_network_credit/        # Neural-network experiment
│   ├── sweep_l2_for_delta.py
│   ├── plot_metric_vs_lambda.py
│   ├── ppnn_experiments.py
│   └── scripts/
│       └── utils_torch.py
├── requirements-nn-credit.txt    # Legacy dependencies for neural_network_credit only
├── pyproject.toml                # Main repository environment
└── README.md
```

Generated results and figures should be written under `out/` and should not be
committed.

## Main environment

The main experiments use the repository-level `pyproject.toml` and `uv.lock`.
For the existing code, use the main environment, for example:

```bash
uv sync
```

The neural-network credit experiment should **not** be run in this environment.
The reason is that the main environment uses a up-to-date Python stack,
including NumPy 2, whereas `whynot` is an old package and depends on the old
`gym==0.21.0` stack.

## Neural-network credit experiment

### Description

The neural-network experiment follows the strategic-classification setting of
Mofakhami, Mitliagkas, and Gidel, *Performative Prediction with Neural Networks*
(AISTATS 2023). It uses the `GiveMeSomeCredit` environment through the `whynot`
package. The performative shift is controlled by a parameter `delta`: after a
negative classification under the previous model, an individual may strategically
modify the manipulable features by copying the corresponding features of another
data point, with probability depending on `delta`.

For each value of `delta`, the code sweeps over the L2 regularization parameter
`lambda`, runs repeated risk minimization for a fixed number of deployments, and
reports the final test accuracy. The qualitative behavior reported in the paper
is that L2 regularization mitigates the accuracy drop caused by the performative
shift, and that the best regularization level increases with the strength of the
performative effect.

### Files

```text
neural_network_credit/
├── sweep_l2_for_delta.py      # Runs one delta value across a grid of lambdas
├── plot_metric_vs_lambda.py   # Builds the final lambda-sweep figure from .pkl outputs
├── ppnn_experiments.py        # Experiment logic, strategic shift, plotting helpers
└── scripts/
    └── utils_torch.py         # Torch models and training/loss utilities
```


### Legacy environment for `whynot`

The `whynot` dependency is the fragile part. Use a separate Python 3.9 environment
and keep packaging tools old enough for `gym==0.21.0`.

Recommended setup:

```bash
python3.9 -m venv .venv-nn-credit
source .venv-nn-credit/bin/activate
python -m pip install --upgrade "pip==23.2.1" "setuptools==65.5.0" "wheel==0.38.4"
python -m pip install -r requirements-nn-credit.txt
```

Check that the credit environment imports correctly:

```bash
python - <<'PY'
import whynot.gym as gym
import torch

env = gym.make("Credit-v0")
data = env.initial_state.values()
print(data["features"].shape, data["labels"].shape)
print(torch.__version__)
PY
```


### Run the sweeps

The paper figure used the following grid:

```text
n_runs        = 2
num_iters     = 3
layers        = 2
learning_rate = 3e-4
test_frac     = 0.9
delta_grid    = [0.1, 0.3, 0.5, 0.7, 0.9]
l2_grid       = [0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
seeds         = 0, 1
```

Run all five sweeps from the repository root:

```bash
mkdir -p out/nn_credit

L2_GRID="0 1e-5 3e-5 1e-4 3e-4 1e-3 3e-3 1e-2"

for DELTA in 0.1 0.3 0.5 0.7 0.9; do
  python neural_network_credit/sweep_l2_for_delta.py \
    --out "out/nn_credit/delta_${DELTA}.pkl" \
    --delta "$DELTA" \
    --n-runs 2 \
    --num-iters 3 \
    --layers 2 \
    --learning-rate 3e-4 \
    --test-frac 0.9 \
    --l2-grid $L2_GRID
done
```

This creates:

```text
out/nn_credit/delta_0.1.pkl
out/nn_credit/delta_0.3.pkl
out/nn_credit/delta_0.5.pkl
out/nn_credit/delta_0.7.pkl
out/nn_credit/delta_0.9.pkl
```

### Plot the figure

From the repository root:

```bash
python neural_network_credit/plot_metric_vs_lambda.py \
  out/nn_credit/delta_0.1.pkl \
  out/nn_credit/delta_0.3.pkl \
  out/nn_credit/delta_0.5.pkl \
  out/nn_credit/delta_0.7.pkl \
  out/nn_credit/delta_0.9.pkl \
  --out-prefix out/nn_credit/metric_vs_lambda_final
```

This writes:

```text
out/nn_credit/metric_vs_lambda_final_acc.pdf
out/nn_credit/metric_vs_lambda_final_loss.pdf
```

The paper uses only:

```text
out/nn_credit/metric_vs_lambda_final_acc.pdf
```

### Relationship with Mofakhami et al. (AISTATS 2023)

The implementation builds on the public code accompanying Mofakhami, Mitliagkas,
and Gidel, *Performative Prediction with Neural Networks*. The retained files are
only the parts needed for the L2-regularization sweep and the final neural-network
figure. Exploratory notebooks, old result folders, local virtual environments,
Python caches, copied simulator files that are not imported by these scripts, and
all stored outputs were intentionally omitted.
