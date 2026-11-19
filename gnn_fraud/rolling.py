from collections import defaultdict, deque

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Batch, Data

from .metrics import evaluate
from .training import predict


def snapshots(graph: Data) -> dict[int, Data]:
    return {
        step: graph.subgraph(graph.timestep == step)
        for step in graph.timestep.unique(sorted=True).tolist()
    }


class ReplayBuffer:
    def __init__(self, nodes_per_class: int = 100):
        self.recent = (deque(maxlen=nodes_per_class), deque(maxlen=nodes_per_class))

    def add(self, step: int, labels: torch.Tensor) -> None:
        for label, recent in enumerate(self.recent):
            indices = (labels == label).nonzero().flatten().tolist()
            recent.extend((step, index) for index in indices)

    def indices_by_step(self) -> dict[int, list[int]]:
        grouped = defaultdict(list)
        for recent in self.recent:
            for step, index in recent:
                grouped[step].append(index)
        return dict(grouped)


def adapt(
    model: nn.Module,
    history: dict[int, Data],
    replay: dict[int, list[int]],
    current: Data,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    epochs: int,
) -> None:
    graphs = []
    for step, indices in replay.items():
        graph = history[step].clone()
        graph.supervised = torch.zeros_like(graph.y, dtype=torch.bool)
        graph.supervised[indices] = True
        graphs.append(graph)
    current = current.clone()
    current.supervised = current.y >= 0
    graphs.append(current)
    batch = Batch.from_data_list(graphs).to(next(model.parameters()).device)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index)
        loss = criterion(logits[batch.supervised], batch.y[batch.supervised])
        loss.backward()
        optimizer.step()


def rolling_predictions(
    model: nn.Module,
    train: Data,
    test: Data,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    epochs: int,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    history = snapshots(train)
    buffer = ReplayBuffer()
    for step, graph in history.items():
        buffer.add(step, graph.y)

    labels_by_step, scores_by_step = [], []
    for step, graph in snapshots(test).items():
        labels, scores = predict(model, graph)
        if not len(labels):
            continue
        metrics = evaluate(labels, scores, threshold)
        print(f"Timestep {step:2d} | {len(labels):4d} labels | macro F1 {metrics['macro_f1']:.4f}")
        labels_by_step.append(labels)
        scores_by_step.append(scores)

        # Labels enter the training history only after this snapshot has been scored.
        adapt(model, history, buffer.indices_by_step(), graph, optimizer, criterion, epochs)
        history[step] = graph
        buffer.add(step, graph.y)

    return np.concatenate(labels_by_step), np.concatenate(scores_by_step)
