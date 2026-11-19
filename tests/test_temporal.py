from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from torch_geometric.data import Data

from gnn_fraud.data import read_elliptic, split_graph
from gnn_fraud.metrics import calibrate_threshold, evaluate
from gnn_fraud.rolling import rolling_predictions
from gnn_fraud.training import fit, predict


def sample_graph():
    return Data(
        x=torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]),
        y=torch.tensor([0, -1, 1, 0, 1, 0]),
        timestep=torch.tensor([29, 29, 30, 31, 35, 36]),
        edge_index=torch.tensor([[0], [1]]),
    )


class DatasetTests(TestCase):
    def test_features_fit_on_training_only(self):
        graph = sample_graph()
        standard = split_graph(graph)
        graph.x[graph.timestep >= 35] *= 1000
        changed = split_graph(graph)
        torch.testing.assert_close(standard.train.x, changed.train.x)
        torch.testing.assert_close(standard.validation.x, changed.validation.x)
        torch.testing.assert_close(changed.train.x.mean(), torch.tensor(0.0), atol=1e-6, rtol=0)
        self.assertEqual(standard.train.timestep.tolist(), [29, 29, 30])
        self.assertEqual(standard.validation.timestep.tolist(), [31])
        self.assertEqual(standard.test.timestep.tolist(), [35, 36])
        self.assertEqual(standard.train.y.tolist(), [0, -1, 1])
        self.assertEqual(standard.train.edge_index.tolist(), [[0], [1]])

    def test_cross_timestep_edge_is_rejected(self):
        graph = sample_graph()
        graph.edge_index = torch.tensor([[0], [4]])
        with self.assertRaisesRegex(ValueError, "within a timestep"):
            split_graph(graph)

    def test_labels_are_joined_by_transaction_id(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            features, edges, classes = (root / name for name in ("features", "edges", "classes"))
            features.write_text("20,1,0\n10,1,0\n", encoding="utf-8")
            classes.write_text("txId,class\n10,1\n20,2\n", encoding="utf-8")
            prepared = MagicMock(raw_paths=(features, edges, classes))
            prepared.__getitem__.return_value = SimpleNamespace(
                x=torch.zeros(2, 1), edge_index=torch.empty(2, 0, dtype=torch.long)
            )
            with patch("gnn_fraud.data.EllipticBitcoinDataset", return_value=prepared):
                graph = read_elliptic(root)
        self.assertEqual(graph.y.tolist(), [0, 1])


class EvaluationTests(TestCase):
    def test_calibration_respects_tied_scores_and_chooses_fewer_alerts(self):
        labels = np.array([1, 0, 1, 0])
        scores = np.array([0.6, 0.6, 0.8, 0.1])
        threshold = calibrate_threshold(labels, scores)
        self.assertEqual(threshold, 0.8)
        self.assertEqual(evaluate(labels, scores, threshold)["confusion_matrix"], [[2, 0], [1, 1]])

    def test_rolling_predicts_before_adapting_and_keeps_calibrated_threshold(self):
        split = split_graph(sample_graph())
        steps = []

        def score(model, graph):
            steps.append(("score", graph.timestep.item()))
            return graph.y.numpy(), np.array([0.6])

        def update(model, history, replay, current, *args):
            step = current.timestep.item()
            steps.append(("update", step))
            self.assertLess(max(history), step)
            self.assertTrue(replay.keys() <= history.keys())
            self.assertEqual(35 in replay, step == 36)

        with (
            patch("gnn_fraud.rolling.predict", side_effect=score),
            patch("gnn_fraud.rolling.adapt", side_effect=update),
            patch("gnn_fraud.rolling.evaluate", wraps=evaluate) as metrics,
        ):
            labels, scores = rolling_predictions(None, split.train, split.test, None, None, 1, 0.75)

        self.assertEqual(steps, [("score", 35), ("update", 35), ("score", 36), ("update", 36)])
        self.assertEqual([call.args[2] for call in metrics.call_args_list], [0.75, 0.75])
        np.testing.assert_array_equal(labels, [1, 0])
        np.testing.assert_allclose(scores, [0.6, 0.6])


class TrainingTests(TestCase):
    def test_best_epoch_restores_weights_and_optimizer(self):
        class Linear(torch.nn.Linear):
            def forward(self, x, edge_index):
                return super().forward(x)

        checkpoints = []
        split = split_graph(sample_graph())
        model = Linear(1, 2)

        def capture(model, graph):
            checkpoints.append(model.weight.detach().clone())
            return predict(model, graph)

        with (
            patch("gnn_fraud.training.predict", side_effect=capture),
            patch("gnn_fraud.training.average_precision_score", side_effect=[0.8, 0.7]),
        ):
            optimizer, _, epoch = fit(model, split, "focal", 2, torch.device("cpu"))

        self.assertEqual(epoch, 1)
        self.assertFalse(torch.equal(checkpoints[0], checkpoints[1]))
        torch.testing.assert_close(model.weight, checkpoints[0])
        self.assertEqual({state["step"].item() for state in optimizer.state.values()}, {1})
