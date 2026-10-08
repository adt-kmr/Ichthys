import argparse
import time

from utils.training_engine import IchthysTrainer, TrainingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Ichthys on multi-camera detection datasets.")
    parser.add_argument("train_folder", type=str, help="Folder containing training scene JSON files.")
    parser.add_argument(
        "val_folder",
        nargs="?",
        default=None,
        help="Optional validation scene folder; do not use the held-out test split for model selection.",
    )
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=400, help="Total number of training epochs.")
    parser.add_argument(
        "--warmup",
        type=int,
        default=100,
        help="Epochs before enabling contrastive and temporal losses.",
    )
    parser.add_argument(
        "--val_every",
        type=int,
        default=999,
        help="Run validation every N epochs. Validation always runs on the final epoch.",
    )
    parser.add_argument(
        "--selection_metric",
        choices=("train", "val"),
        default="train",
        help=(
            "Metric used for best_ckpt.pth. The reported models used 'train'; "
            "use 'val' to select only on validation epochs."
        ),
    )
    parser.add_argument("--debug", action="store_true", help="Enable per-frame debug logging to file.")
    parser.add_argument(
        "--debug_file",
        type=str,
        default=None,
        help="Path to append debug logs. Defaults to <checkpoint_dir>/debug.log.",
    )
    parser.add_argument(
        "--use_dino",
        action="store_true",
        default=True,
        help="Use DINO visual embeddings and train the encoder end-to-end.",
    )
    parser.add_argument(
        "--no_dino",
        action="store_false",
        dest="use_dino",
        help="Disable DINO and train with geometric features only.",
    )
    parser.add_argument(
        "--no_global_attention",
        action="store_true",
        help="Replace global self-attention with the reported tokenwise MLP ablation.",
    )
    parser.add_argument(
        "--dino_repo_dir",
        type=str,
        default="dinov3",
        help="DINOv3 checkout path (relative paths resolve from the repository root).",
    )
    parser.add_argument(
        "--dino_weights",
        type=str,
        default="DINOV3Model/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
        help="DINOv3 weights path (relative paths resolve from the repository root).",
    )
    parser.add_argument(
        "--greedygroup",
        action="store_true",
        help="Use greedy matching inside grouping instead of Hungarian matching.",
    )
    parser.add_argument(
        "--linearsumgroup",
        action="store_true",
        help="Force Hungarian matching inside grouping.",
    )
    parser.add_argument("--train_workers", type=int, default=64, help="Parallel workers for training JSON parsing.")
    parser.add_argument("--val_workers", type=int, default=32, help="Parallel workers for validation JSON parsing.")
    return parser.parse_args()


def resolve_grouping_mode(args: argparse.Namespace) -> str:
    if getattr(args, "linearsumgroup", False):
        return "hungarian"
    if getattr(args, "greedygroup", False):
        return "greedy"
    return "hungarian"


def main() -> None:
    args = parse_args()
    config = TrainingConfig(
        train_folder=args.train_folder,
        val_folder=args.val_folder,
        checkpoint_dir=args.checkpoint_dir,
        total_epochs=args.epochs,
        warmup_epochs=args.warmup,
        debug=args.debug,
        debug_file=args.debug_file,
        val_every=args.val_every,
        selection_metric=args.selection_metric,
        use_dino=args.use_dino,
        no_global_attention=args.no_global_attention,
        dino_repo_dir=args.dino_repo_dir,
        dino_weights=args.dino_weights,
        grouping_match_mode=resolve_grouping_mode(args),
        train_workers=args.train_workers,
        val_workers=args.val_workers,
    )

    start_time = time.time()
    trainer = IchthysTrainer(config)
    trainer.run()
    elapsed_hours = (time.time() - start_time) / 3600.0
    print(f"Total training time: {elapsed_hours:.2f} hours")


if __name__ == "__main__":
    main()