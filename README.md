# GNN Fraud Detection

This repository implements a highly scalable Graph Neural Network (GNN) pipeline to detect illicit financial transactions (e.g., money laundering, fraud) in the Elliptic Bitcoin dataset. 

By modeling financial transactions as a graph and employing relational inductive biases, this project demonstrates how graph-based learning significantly outperforms traditional tabular machine learning (like XGBoost) in identifying obfuscated fraud patterns.

## Advanced Features
- **Temporal Graph Processing:** Strict chronological train/test splitting across 49 timesteps to prevent time-travel data leakage.
- **Continual Learning (Rolling Window):** Simulates real-world production by incrementally fine-tuning the model through future timesteps using a **Historical Replay Buffer** to combat concept drift.
- **Threshold Calibration:** Sweeps validation probabilities to dynamically calibrate the decision boundary rather than relying on a naïve 0.5 threshold, heavily suppressing alert fatigue.
- **Multiple GNN Architectures:** Implementations of **GraphSAGE**, **GCN**, and **GAT** using PyTorch Geometric.
- **Class Imbalance Mitigation:** Integrated custom **Focal Loss** ($\alpha=0.85$, $\gamma=2.0$) to handle extreme minority classes (~2% illicit nodes).

## Benchmark Results (Rolling Window + Calibration)

To combat the degradation of static models over time (Concept Drift), the models were evaluated using the Rolling Window Incremental Trainer alongside Threshold Calibration. 

**GraphSAGE** emerged as a production-grade architecture, slashing the False Positive Rate to under 1% while maximizing overall accuracy.

| Metric                 | GraphSAGE (Best) | GCN      | GAT      |
| :--------------------- | :--------------- | :------- | :------- |
| **Macro F1 Score**     | **0.8308**       | 0.6787   | 0.8084   |
| **PR-AUC**             | **0.7159**       | 0.4043   | 0.6706   |
| **Recall (Illicit)**   | **0.5753**       | 0.2918   | 0.5596   |
| **False Positive Rate**| **0.0081**       | 0.0146   | 0.0134   |
| **Threshold Used**     | 0.73             | 0.67     | 0.66     |

## Installation

1. Clone this repository.
2. Install the required dependencies:
```bash
pip install torch torch_geometric scikit-learn matplotlib seaborn numpy pandas
```

## Usage

The dataset (Elliptic Bitcoin Dataset) is automatically downloaded and processed on the first run via PyTorch Geometric.

You can train and evaluate the models using the CLI entry point `main.py`.

### 1. Rolling Window Training (Recommended)
Train incrementally on the test set (timesteps 35-49) using the Replay Buffer and calibrate the optimal decision threshold:
```bash
python main.py --model GraphSAGE --epochs 100 --calibrate --rolling
```

### 2. Static Training
Train a standard static model on timesteps 1-34 and evaluate directly on timesteps 35-49:
```bash
python main.py --model GAT --epochs 150 --calibrate
```

## Repository Structure
- `main.py`: CLI entry point for training and evaluation.
- `dataset.py`: Handles dataset downloading, temporal splitting, feature normalization, and timestep extraction.
- `rolling_window.py`: Implements Continual Learning loops and the Replay Buffer.
- `models.py`: Defines GraphSAGE, GCN, and GAT architectures.
- `losses.py`: Implements custom Focal Loss.
- `train.py`: Standard PyTorch static training loop.
- `evaluate.py`: Calculates metrics, calibrates thresholds, and generates confusion matrices.
