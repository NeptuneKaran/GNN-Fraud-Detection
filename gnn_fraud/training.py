from copy import deepcopy

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch_geometric.data import Data

from .data import TemporalSplit
from .losses import FocalLoss, get_weighted_ce_loss


@torch.inference_mode()
def predict(model: nn.Module, graph: Data) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    device = next(model.parameters()).device
    mask = graph.y >= 0
    logits = model(graph.x.to(device), graph.edge_index.to(device))
    scores = logits[mask.to(device)].softmax(dim=-1)[:, 1]
    return graph.y[mask].cpu().numpy(), scores.cpu().numpy()


def fit(
    model: nn.Module,
    data: TemporalSplit,
    loss_name: str,
    epochs: int,
    device: torch.device,
) -> tuple[torch.optim.Optimizer, nn.Module, int]:
    model.to(device)
    train = data.train.clone().to(device)
    validation = data.validation.clone().to(device)
    mask = train.y >= 0
    labels = train.y[mask]
    criterion = (
        FocalLoss(alpha=0.85, gamma=2.0)
        if loss_name == "focal"
        else get_weighted_ce_loss(labels)
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=0.0001)

    best_ap = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(train.x, train.edge_index)
        loss = criterion(logits[mask], labels)
        loss.backward()
        optimizer.step()

        val_labels, val_scores = predict(model, validation)
        val_ap = average_precision_score(val_labels, val_scores)
        if val_ap > best_ap:
            best_ap = val_ap
            best_epoch = epoch
            best_model = deepcopy(model.state_dict())
            best_optimizer = deepcopy(optimizer.state_dict())
        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | loss {loss.item():.4f} | validation AP {val_ap:.4f}")

    model.load_state_dict(best_model)
    optimizer.load_state_dict(best_optimizer)
    print(f"Selected epoch {best_epoch} (validation AP {best_ap:.4f})")
    return optimizer, criterion, best_epoch
