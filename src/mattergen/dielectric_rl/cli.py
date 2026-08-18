from __future__ import annotations

import argparse
import json

from omegaconf import OmegaConf

from mattergen.dielectric_rl import DEFAULT_CONFIG_PATH
from mattergen.dielectric_rl.loop import prepare_reward_table_and_buffer, train_from_replay_buffer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-property dielectric RL utilities for MatterGen."
    )
    parser.add_argument(
        "command",
        choices=["prepare", "train"],
        help="prepare builds the reward table and replay buffer; train also runs offline RL fine-tuning.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to a YAML config file.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="OmegaConf dotlist override, e.g. reward.target_dielectric_scalar=8.0",
    )
    parser.add_argument(
        "--topk_ratio",
        type=float,
        default=None,
        help="Fraction of eval_size to keep for MatInvent-style top-k selection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = OmegaConf.load(args.config)
    if args.override:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.override))
    if "training" not in config:
        config.training = OmegaConf.create()
    if args.topk_ratio is not None:
        config.training.topk_ratio = float(args.topk_ratio)

    if args.command == "prepare":
        result = prepare_reward_table_and_buffer(config)
        print(json.dumps(result["summary"], indent=2))
        return

    if not bool(config.training.enabled):
        config.training.enabled = True
    result = train_from_replay_buffer(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
