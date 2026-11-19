# Elliptic transaction classification

GraphSAGE, GCN, and GAT classify Bitcoin transactions as licit or illicit using the Elliptic graph. A class-weighted XGBoost model provides a tabular baseline.

## Run

From the repository root, in a Python 3.10+ environment:

```bash
python -m pip install -r requirements.txt
python main.py --model all --device cpu
```

PyTorch Geometric downloads the public dataset into `data/` on the first run. Use `--data-dir` to point at an existing Elliptic cache. Each run writes metrics and a confusion matrix to `runs/` by default. Use `--output-dir` to keep experiments with different settings.

```bash
python main.py --model GraphSAGE --loss weighted_ce --device cpu --output-dir runs/weighted
python main.py --model GraphSAGE --rolling --device cpu
python -m unittest discover -s tests -v
```

The default GNN settings are 100 initial epochs, focal loss, seed 42, and four CPU threads. Calibration is enabled by default. `--no-calibrate` uses 0.50. The same options are available through `python -m gnn_fraud` or the installed `gnn-fraud` command. `--model all --rolling` runs the three GNNs.

## Evaluation protocol

| Split | Timesteps | Labeled transactions |
| --- | --- | ---: |
| Training | 1–30 | 26,905 |
| Validation | 31–34 | 2,989 |
| Test | 35–49 | 16,670 |

The scaler and class weights use the training split. The GNN checkpoint is selected by validation average precision, and the classification threshold is selected by validation Macro F1. Unlabeled nodes remain in their own timestep graphs as neighbors.

In static mode the model scores the entire held-out period without updating. In rolling mode the GNN scores each timestep before using its labels for ten adaptation epochs with replay. Both modes keep the validation cutoff fixed during testing.

## Benchmark

Fresh CPU runs with seed 42 and default settings. Each model uses its own validation cutoff.

| Static model | Macro F1 | False-positive rate | Illicit recall | PR-AUC |
| --- | ---: | ---: | ---: | ---: |
| GraphSAGE | 0.7313 | 4.72% | 56.23% | 0.3436 |
| GCN | 0.7148 | 3.79% | 47.09% | 0.3101 |
| GAT | 0.6992 | 3.54% | 42.11% | 0.3799 |
| XGBoost | 0.8781 | 1.08% | 72.48% | 0.7886 |

GraphSAGE had the highest Macro F1 and recall of the static GNNs. GCN and GAT produced fewer false alerts, and XGBoost led the static comparison overall.

GraphSAGE with weighted cross-entropy reached 0.7222 Macro F1 and 4.39% false-positive rate. With focal loss and rolling updates it reached 0.8135 Macro F1, 3.98% false-positive rate, and 76.64% recall. The rolling run assumes that labels from a scored timestep are available before the next one.

## Code map

| File | Responsibility |
| --- | --- |
| [`gnn_fraud/cli.py`](gnn_fraud/cli.py) | Experiment settings and result files |
| [`gnn_fraud/data.py`](gnn_fraud/data.py) | Transaction labels, temporal splits, and feature scaling |
| [`gnn_fraud/models.py`](gnn_fraud/models.py), [`losses.py`](gnn_fraud/losses.py) | GNN architectures and imbalance handling |
| [`gnn_fraud/training.py`](gnn_fraud/training.py), [`rolling.py`](gnn_fraud/rolling.py) | Initial training and online adaptation |
| [`gnn_fraud/metrics.py`](gnn_fraud/metrics.py), [`baseline.py`](gnn_fraud/baseline.py) | Threshold selection, reporting, and XGBoost |
