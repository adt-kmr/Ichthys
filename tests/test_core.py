from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from evaluate import _benchmark_dist_thr, _euclidean_cost_matrix, _read_gt_csv, _read_pred_txt
from sort3d import run_sort
from train import parse_args
from utils.association_module_noattn import AssociationNoGlobalAttention
from utils.generate_pseudogt import make_pseudo_gt
from utils.training_engine import EpochMetrics, IchthysTrainer, TrainingConfig


class EvaluationTests(unittest.TestCase):
    def test_benchmark_matching_gate_is_selected_from_gt_filename(self) -> None:
        for index in range(1, 7):
            self.assertEqual(_benchmark_dist_thr(Path(f"test{index}-gt.csv")), 1.0)
        for name in ("zebra02-new.csv", "zebra04.csv"):
            self.assertEqual(_benchmark_dist_thr(Path(name)), 0.5)
        with self.assertRaisesRegex(ValueError, "Unknown benchmark GT filename"):
            _benchmark_dist_thr(Path("custom.csv"))

    def test_distance_gate_masks_far_pairs(self) -> None:
        gt = [("fish-1", np.array([0.0, 0.0, 0.0], dtype=np.float32))]
        predictions = [
            ("A", np.array([0.5, 0.0, 0.0], dtype=np.float32)),
            ("B", np.array([2.0, 0.0, 0.0], dtype=np.float32)),
        ]
        costs = _euclidean_cost_matrix(gt, predictions, dist_thr=1.0)
        self.assertAlmostEqual(float(costs[0, 0]), 0.5)
        self.assertTrue(np.isnan(costs[0, 1]))

    def test_track_file_parsers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pred = root / "pred.txt"
            gt = root / "gt.csv"
            pred.write_text(
                "object,Timestamp,X,Y,Z,group\nA,1,1.0,2.0,3.0,{cam0-0}\n",
                encoding="utf-8",
            )
            gt.write_text(
                "Actor,Timestamp,X,Y,Z\nfish-1,1,1.0,2.0,3.0\n",
                encoding="utf-8",
            )
            self.assertEqual(_read_pred_txt(pred)[1][0][0], "A")
            self.assertEqual(_read_gt_csv(gt)[1][0][0], "fish-1")


class PseudoLabelTests(unittest.TestCase):
    def test_reprojection_containment_produces_positive_pair(self) -> None:
        intrinsic = np.eye(3)
        camera_0 = np.hstack([np.eye(3), np.zeros((3, 1))])
        camera_1 = np.hstack([np.eye(3), np.array([[-1.0], [0.0], [0.0]])])
        cameras = {
            "cam0-K": intrinsic,
            "cam0-R|T": camera_0,
            "cam1-K": intrinsic,
            "cam1-R|T": camera_1,
        }
        detections = [
            {"cam_id": 0, "segmentation-centroid": [0.0, 0.0], "bbox": [-0.1, -0.1, 0.1, 0.1]},
            {"cam_id": 1, "segmentation-centroid": [-0.2, 0.0], "bbox": [-0.3, -0.1, -0.1, 0.1]},
        ]
        labels, confidence = make_pseudo_gt(detections, cameras)
        self.assertEqual(int(labels[0, 1]), 1)
        self.assertGreater(float(confidence[0, 1]), 0.99)


class NoGlobalAttentionTests(unittest.TestCase):
    def test_unrelated_token_does_not_change_existing_pair_scores(self) -> None:
        torch.manual_seed(0)
        model = AssociationNoGlobalAttention(
            in_dim=4,
            d_model=8,
            num_layers=2,
            dim_ff=16,
            dropout=0.0,
            use_rope=False,
        ).eval()
        features = torch.randn(3, 4)
        camera_ids = torch.tensor([0, 1, 2])

        original = model(features, camera_ids)
        changed_features = features.clone()
        changed_features[2] += 100.0
        changed = model(changed_features, camera_ids)

        torch.testing.assert_close(original[:2, :2], changed[:2, :2])


class CheckpointSelectionTests(unittest.TestCase):
    def test_training_cli_does_not_require_held_out_split(self) -> None:
        with patch.object(sys, "argv", ["train.py", "train"]):
            args = parse_args()
        self.assertEqual(args.train_folder, "train")
        self.assertIsNone(args.val_folder)
        with patch.object(sys, "argv", ["train.py", "train", "validation"]):
            args = parse_args()
        self.assertEqual(args.val_folder, "validation")

    def test_validation_selection_requires_validation_data(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires a separate validation folder"):
            TrainingConfig(train_folder="train", selection_metric="val")

    def test_train_only_does_not_load_or_cache_validation_data(self) -> None:
        trainer = object.__new__(IchthysTrainer)
        trainer.config = TrainingConfig(train_folder="train", train_workers=1)
        with patch("utils.training_engine.load_and_parse_json_folder", return_value=[{"names": "train1"}]) as loader, \
                patch("utils.training_engine.build_pseudo_gt_cache_for_dataset") as cache:
            train_dataset, val_dataset = trainer._load_datasets()
            trainer._build_pseudo_gt_cache(train_dataset, val_dataset)
        self.assertIsNone(val_dataset)
        loader.assert_called_once_with("train", num_workers=1)
        cache.assert_called_once_with(dataset=train_dataset, dataset_root="train", name="train")

    @staticmethod
    def _trainer(root: Path, metric: str) -> IchthysTrainer:
        trainer = object.__new__(IchthysTrainer)
        trainer.config = TrainingConfig(
            train_folder="train",
            val_folder="val",
            checkpoint_dir=str(root),
            selection_metric=metric,
        )
        trainer.checkpoint_dir = root
        trainer.best_score = float("inf")
        trainer._checkpoint_payload = lambda epoch, train_avg, val_avg: {"epoch": epoch}
        return trainer

    def test_reported_protocol_selects_on_training_loss(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, patch("torch.save") as save:
            trainer = self._trainer(Path(temp_dir), "train")
            trainer._save_checkpoints(
                epoch=1,
                train_metrics=EpochMetrics(total_loss=2.0, steps=2),
                val_metrics=None,
                validation_ran=False,
            )
            self.assertEqual(trainer.best_score, 1.0)
            save.assert_called_once()

    def test_validation_selection_waits_for_validation_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, patch("torch.save") as save:
            trainer = self._trainer(Path(temp_dir), "val")
            trainer._save_checkpoints(
                epoch=1,
                train_metrics=EpochMetrics(total_loss=2.0, steps=2),
                val_metrics=None,
                validation_ran=False,
            )
            self.assertEqual(trainer.best_score, float("inf"))
            save.assert_not_called()


class Sort3DTests(unittest.TestCase):
    def test_run_sort_preserves_single_track_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            detections = root / "detections.csv"
            output = root / "tracks.csv"
            detections.write_text(
                "object,Timestamp,X,Y,Z,group\n"
                "det0,1,0.0,0.0,0.0,{}\n"
                "det1,2,0.1,0.0,0.0,{}\n",
                encoding="utf-8",
            )

            run_sort(
                input_path=detections,
                output_path=output,
                gate=1.0,
                max_age=2,
                min_hits=1,
                process_noise=0.01,
                measurement_noise=0.05,
                fill_misses=False,
            )

            rows = output.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(rows), 3)
            self.assertTrue(rows[1].startswith("A,1,"))
            self.assertTrue(rows[2].startswith("A,2,"))


if __name__ == "__main__":
    unittest.main()
