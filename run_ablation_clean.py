import argparse
import json
import os
import shlex
import subprocess
import sys
from typing import Dict, List, Sequence


def build_parser():
    parser = argparse.ArgumentParser(description="Run clean baseline ablations for nested memory and TTA")
    parser.add_argument("--file-path", default="datasets/Kvasir/train")
    parser.add_argument("--test-file-path", default="datasets/Kvasir/test")
    parser.add_argument("--save-root", default="outputs/kvasir_clean_ablation")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--mode", choices=["plan", "run"], default="plan")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--variants", nargs="+", choices=["baseline", "slow_only", "fast_slow"], default=["baseline", "slow_only", "fast_slow"])
    parser.add_argument("--eval-checkpoint", default="")
    parser.add_argument("--image-size", type=int, nargs=2, default=(384, 384), metavar=("H", "W"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--encoder-name", choices=["tiny_convnext", "convnext_tiny", "convnext_small"], default="convnext_small")
    parser.add_argument("--use-pretrained", action="store_true")
    parser.add_argument("--decoder-channels", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    parser.add_argument("--decoder-lr", type=float, default=8e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--small-polyp-sampling-power", type=float, default=0.35)
    parser.add_argument("--nested-start-epoch", type=int, default=8)
    parser.add_argument("--nested-dim", type=int, default=128)
    parser.add_argument("--nested-prototypes", type=int, default=8)
    parser.add_argument("--nested-residual-scale", type=float, default=0.05)
    parser.add_argument("--nested-max-norm", type=float, default=1.0)
    parser.add_argument("--nested-memory-hidden", type=int, default=128)
    parser.add_argument("--nested-slow-momentum-scale", type=float, default=0.25)
    parser.add_argument("--nested-momentum", type=float, default=0.03)
    parser.add_argument("--nested-skip-margin", type=float, default=0.002)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.35, 0.40, 0.45, 0.50, 0.55, 0.60])
    parser.add_argument("--flip-tta-scales", type=float, nargs="+", default=[1.0])
    parser.add_argument("--ms-tta-scales", type=float, nargs="+", default=[1.0, 0.875, 1.125])
    return parser


def _command_to_string(command: Sequence[str]) -> str:
    return shlex.join(list(command))


def _repo_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _resolve_path(path: str, repo_root: str) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(repo_root, path))


def _validate_existing_path(path: str, label: str):
    if path and not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found: {path}")


def _base_train_command(args, repo_root: str, save_root: str) -> List[str]:
    command = [
        args.python_bin,
        os.path.join(repo_root, "train_baseline_clean.py"),
        "--file-path",
        args.file_path,
        "--save-root",
        save_root,
        "--image-size",
        str(args.image_size[0]),
        str(args.image_size[1]),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--epochs",
        str(args.epochs),
        "--warmup-epochs",
        str(args.warmup_epochs),
        "--seed",
        str(args.seed),
        "--encoder-name",
        args.encoder_name,
        "--decoder-channels",
        str(args.decoder_channels),
        "--dropout",
        str(args.dropout),
        "--encoder-lr",
        str(args.encoder_lr),
        "--decoder-lr",
        str(args.decoder_lr),
        "--weight-decay",
        str(args.weight_decay),
        "--nested-start-epoch",
        str(args.nested_start_epoch),
        "--nested-dim",
        str(args.nested_dim),
        "--nested-prototypes",
        str(args.nested_prototypes),
        "--nested-residual-scale",
        str(args.nested_residual_scale),
        "--nested-max-norm",
        str(args.nested_max_norm),
        "--nested-memory-hidden",
        str(args.nested_memory_hidden),
        "--nested-slow-momentum-scale",
        str(args.nested_slow_momentum_scale),
        "--nested-momentum",
        str(args.nested_momentum),
        "--nested-skip-margin",
        str(args.nested_skip_margin),
        "--small-polyp-sampling-power",
        str(args.small_polyp_sampling_power),
        "--thresholds",
        *[str(threshold) for threshold in args.thresholds],
        "--stratified-split",
    ]
    if args.use_pretrained:
        command.append("--use-pretrained")
    if args.use_ema:
        command.append("--use-ema")
    if args.init_checkpoint:
        command.extend(["--init-checkpoint", args.init_checkpoint])
    return command


def _variant_train_command(args, repo_root: str, variant: str, save_root: str) -> List[str]:
    command = _base_train_command(args, repo_root, save_root)
    if variant == "baseline":
        command.extend(["--eval-nested-mode", "off", "--test-nested-mode", "off"])
        return command
    if variant == "slow_only":
        command.extend(
            [
                "--enable-nested",
                "--nested-memory-mode",
                "slow_only",
                "--eval-nested-mode",
                "auto",
                "--test-nested-mode",
                "auto",
            ]
        )
        return command
    if variant == "fast_slow":
        command.extend(
            [
                "--enable-nested",
                "--nested-memory-mode",
                "fast_slow",
                "--eval-nested-mode",
                "auto",
                "--test-nested-mode",
                "auto",
            ]
        )
        return command
    raise ValueError(f"Unsupported ablation variant: {variant}")


def _eval_commands(args, repo_root: str, checkpoint: str, save_root: str) -> Dict[str, List[str]]:
    return {
        "no_tta_nested_auto": [
            args.python_bin,
            os.path.join(repo_root, "test_baseline.py"),
            "--file-path",
            args.test_file_path,
            "--checkpoint",
            checkpoint,
            "--save-root",
            os.path.join(save_root, "no_tta_nested_auto"),
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--no-use-tta",
        ],
        "flip_tta_nested_auto": [
            args.python_bin,
            os.path.join(repo_root, "test_baseline.py"),
            "--file-path",
            args.test_file_path,
            "--checkpoint",
            checkpoint,
            "--save-root",
            os.path.join(save_root, "flip_tta_nested_auto"),
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--use-tta",
            "--tta-scales",
            *[str(scale) for scale in args.flip_tta_scales],
        ],
        "ms_tta_nested_auto": [
            args.python_bin,
            os.path.join(repo_root, "test_baseline.py"),
            "--file-path",
            args.test_file_path,
            "--checkpoint",
            checkpoint,
            "--save-root",
            os.path.join(save_root, "ms_tta_nested_auto"),
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--use-tta",
            "--tta-scales",
            *[str(scale) for scale in args.ms_tta_scales],
        ],
        "ms_tta_nested_off": [
            args.python_bin,
            os.path.join(repo_root, "test_baseline.py"),
            "--file-path",
            args.test_file_path,
            "--checkpoint",
            checkpoint,
            "--save-root",
            os.path.join(save_root, "ms_tta_nested_off"),
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--use-tta",
            "--tta-scales",
            *[str(scale) for scale in args.ms_tta_scales],
            "--no-use-nested",
        ],
    }


def _run_command(command: Sequence[str], cwd: str):
    subprocess.run(list(command), check=True, cwd=cwd)


def main():
    args = build_parser().parse_args()
    repo_root = _repo_root()
    args.file_path = _resolve_path(args.file_path, repo_root)
    args.test_file_path = _resolve_path(args.test_file_path, repo_root)
    args.save_root = _resolve_path(args.save_root, repo_root)
    args.eval_checkpoint = _resolve_path(args.eval_checkpoint, repo_root) if args.eval_checkpoint else ""
    args.init_checkpoint = _resolve_path(args.init_checkpoint, repo_root) if args.init_checkpoint else ""

    _validate_existing_path(args.file_path, "Train file path")
    _validate_existing_path(args.test_file_path, "Test file path")
    if args.eval_checkpoint:
        _validate_existing_path(args.eval_checkpoint, "Eval checkpoint")
    if args.init_checkpoint:
        _validate_existing_path(args.init_checkpoint, "Init checkpoint")

    os.makedirs(args.save_root, exist_ok=True)

    manifest = {
        "mode": args.mode,
        "variants": {},
        "standalone_eval": {},
    }

    if args.eval_checkpoint:
        eval_root = os.path.join(args.save_root, "standalone_eval")
        os.makedirs(eval_root, exist_ok=True)
        eval_commands = _eval_commands(args, repo_root=repo_root, checkpoint=args.eval_checkpoint, save_root=eval_root)
        manifest["standalone_eval"] = {name: _command_to_string(command) for name, command in eval_commands.items()}
        if args.mode == "run":
            for command in eval_commands.values():
                _run_command(command, cwd=repo_root)

    if not args.skip_train:
        for variant in args.variants:
            variant_root = os.path.join(args.save_root, variant)
            train_command = _variant_train_command(args, repo_root=repo_root, variant=variant, save_root=variant_root)
            checkpoint_path = os.path.join(variant_root, "best_model.pth")
            eval_root = os.path.join(variant_root, "eval_matrix")
            eval_commands = _eval_commands(args, repo_root=repo_root, checkpoint=checkpoint_path, save_root=eval_root)

            manifest["variants"][variant] = {
                "train": _command_to_string(train_command),
                "checkpoint": checkpoint_path,
                "eval": {name: _command_to_string(command) for name, command in eval_commands.items()},
            }

            if args.mode == "run":
                _run_command(train_command, cwd=repo_root)
                for command in eval_commands.values():
                    _run_command(command, cwd=repo_root)

    with open(os.path.join(args.save_root, "ablation_plan.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
