from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.datasets import EllipticBitcoinDataset


@dataclass
class TemporalSplit:
    train: Data
    validation: Data
    test: Data


def read_elliptic(root: Path) -> Data:
    dataset = EllipticBitcoinDataset(root=str(root))
    graph = dataset[0]
    transactions = pd.read_csv(dataset.raw_paths[0], header=None, usecols=[0, 1])
    classes = pd.read_csv(dataset.raw_paths[2], dtype={"class": str})
    labels = classes.set_index("txId")["class"].map({"2": 0, "1": 1, "unknown": -1})
    labels = labels.reindex(transactions[0])
    if labels.isna().any():
        raise ValueError("Every transaction must have a recognized label or be marked unknown.")

    return Data(
        x=graph.x,
        edge_index=graph.edge_index,
        y=torch.from_numpy(labels.to_numpy(dtype="int64", copy=True)),
        timestep=torch.from_numpy(transactions[1].to_numpy(dtype="int64", copy=True)),
    )


def split_graph(graph: Data) -> TemporalSplit:
    source, target = graph.edge_index
    if (graph.timestep[source] != graph.timestep[target]).any():
        raise ValueError("Expected Elliptic edges to stay within a timestep.")

    train = graph.subgraph(graph.timestep <= 30)
    validation = graph.subgraph((graph.timestep >= 31) & (graph.timestep <= 34))
    test = graph.subgraph(graph.timestep >= 35)

    # Elliptic features include heavy-tailed transaction amounts and counts.
    for part in (train, validation, test):
        part.x = part.x.sign() * part.x.abs().log1p()
    scaler = StandardScaler().fit(train.x.numpy())
    for part in (train, validation, test):
        part.x = torch.from_numpy(scaler.transform(part.x.numpy()))
    return TemporalSplit(train, validation, test)


def load_data(root: Path) -> TemporalSplit:
    return split_graph(read_elliptic(root))
