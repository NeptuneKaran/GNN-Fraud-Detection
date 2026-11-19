from torch_geometric.data import Data
from xgboost import XGBClassifier


def tabular_data(graph: Data):
    mask = graph.y >= 0
    return graph.x[mask].numpy(), graph.y[mask].numpy()


def fit_xgboost(train: Data, seed: int, threads: int) -> XGBClassifier:
    features, labels = tabular_data(train)
    positives = (labels == 1).sum()
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=float((len(labels) - positives) / positives),
        tree_method="hist",
        random_state=seed,
        n_jobs=threads,
    )
    model.fit(features, labels)
    return model


def predict_xgboost(model: XGBClassifier, graph: Data):
    features, labels = tabular_data(graph)
    return labels, model.predict_proba(features)[:, 1]
