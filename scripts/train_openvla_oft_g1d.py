#!/usr/bin/env python3
"""Fine-tune OpenVLA-OFT on audited G1-D Family Home Expert trajectories."""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OFT_ROOT = ROOT / "third_party/openvla-oft"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OFT_ROOT))

from scripts.g1d_openvla_oft_data import G1DOFTDataset, build_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demo-dir",
        type=Path,
        default=ROOT / "outputs/family_home_vln/expert_demos_head_clip01",
    )
    parser.add_argument("--manifest", type=Path, default=ROOT / "outputs/family_home_vln/openvla_oft/manifest.json")
    parser.add_argument("--model", type=str, default=str(ROOT / "checkpoints/openvla-7b"))
    parser.add_argument("--output", type=Path, default=ROOT / "checkpoints/openvla-oft-g1d")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--save-freq", type=int, default=250)
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument("--data-smoke", action="store_true")
    return parser.parse_args()


def _load_official_finetuner():
    path = OFT_ROOT / "vla-scripts/finetune.py"
    spec = importlib.util.spec_from_file_location("openvla_oft_finetune_g1d", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load OFT finetuner at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    args = parse_args()
    args.demo_dir = args.demo_dir.resolve()
    args.manifest = args.manifest.resolve()
    model_path = Path(args.model).expanduser()
    if model_path.exists():
        args.model = str(model_path.resolve())
    args.output = args.output.resolve()
    manifest = build_manifest(args.demo_dir, args.manifest)
    print(
        f"Audited {manifest['episode_count']} episodes, {manifest['frame_count']} frames, "
        f"{manifest['sample_count']} eight-action samples"
    )
    if args.data_only:
        return

    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    module = _load_official_finetuner()

    if args.data_smoke:
        from torch.utils.data import DataLoader
        from transformers import AutoProcessor

        from prismatic.models.backbones.llm.prompting import PurePromptBuilder
        from prismatic.util.data_utils import PaddedCollatorForActionPrediction
        from prismatic.vla.action_tokenizer import ActionTokenizer
        from prismatic.vla.datasets import RLDSBatchTransform

        processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
        transform = RLDSBatchTransform(
            ActionTokenizer(processor.tokenizer),
            processor.tokenizer,
            image_transform=processor.image_processor.apply_transform,
            prompt_builder_fn=PurePromptBuilder,
            use_wrist_image=False,
            use_proprio=False,
        )
        dataset = G1DOFTDataset(args.manifest, transform)
        collator = PaddedCollatorForActionPrediction(
            processor.tokenizer.model_max_length,
            processor.tokenizer.pad_token_id,
            padding_side="right",
        )
        batch = next(iter(DataLoader(dataset, batch_size=1, collate_fn=collator)))
        expected = (1, 8, 7)
        if tuple(batch["actions"].shape) != expected:
            raise RuntimeError(
                f"invalid OFT action batch {tuple(batch['actions'].shape)} != {expected}"
            )
        print(
            "OFT data smoke passed: "
            f"pixels={tuple(batch['pixel_values'].shape)}, "
            f"tokens={tuple(batch['input_ids'].shape)}, "
            f"actions={tuple(batch['actions'].shape)}"
        )
        return

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    samples_per_pass = int(manifest["sample_count"])
    samples_needed = (
        (args.max_steps + 2)
        * args.batch_size
        * args.grad_accumulation
        * world_size
    )
    repeat = max(1, math.ceil(samples_needed / samples_per_pass))

    class FamilyHomeDataset(G1DOFTDataset):
        def __init__(self, _root, _name, batch_transform, **_kwargs):
            super().__init__(args.manifest, batch_transform, repeat=repeat)

    module.RLDSDataset = FamilyHomeDataset

    # The official OFT loop uses DDP. Enable activation checkpointing before
    # wrapping the VLA so a 24 GB RTX 4090 can run batch size 1 reliably.
    original_wrap_ddp = module.wrap_ddp

    def checkpointed_wrap_ddp(model, device_id, find_unused=False):
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if getattr(model, "config", None) is not None:
            model.config.use_cache = False
        wrapped = original_wrap_ddp(model, device_id, find_unused)
        # LoRA parameters participate in re-entrant activation-checkpoint
        # backward passes.  Declaring the graph static prevents DDP from
        # registering the same adapter parameter as ready twice.
        wrapped._set_static_graph()
        return wrapped

    module.wrap_ddp = checkpointed_wrap_ddp
    cfg = module.FinetuneConfig(
        vla_path=args.model,
        data_root_dir=args.manifest.parent,
        dataset_name="g1d_family_home_pick",
        run_root_dir=args.output,
        shuffle_buffer_size=512,
        use_l1_regression=True,
        use_diffusion=False,
        num_images_in_input=1,
        use_proprio=False,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        grad_accumulation_steps=args.grad_accumulation,
        max_steps=args.max_steps,
        use_val_set=False,
        save_freq=min(args.save_freq, args.max_steps),
        save_latest_checkpoint_only=True,
        image_aug=True,
        use_lora=True,
        lora_rank=args.lora_rank,
        merge_lora_during_training=False,
        wandb_entity="disabled",
        wandb_project="g1d-openvla-oft",
        wandb_log_freq=1,
        run_id_override="g1d-family-home-oft",
    )
    # OFT's model-logic synchronizer searches from the current directory.
    # Run it from the OFT checkout so the checkpoint receives the OFT-aware
    # configuration/modeling files rather than retaining vanilla OpenVLA code.
    os.chdir(OFT_ROOT)
    # Bypass Draccus' CLI wrapper because this launcher has already parsed
    # and validated its G1-D-specific arguments above.
    module.finetune.__wrapped__(cfg)


if __name__ == "__main__":
    main()
