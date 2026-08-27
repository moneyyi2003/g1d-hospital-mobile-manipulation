#!/usr/bin/env python3
"""LoRA fine-tune OpenVLA-7B on G1-D grasp demonstrations.

OpenVLA-7B architecture
------------------------
- Vision encoder: **DINOv2 + SigLIP fused** backbone → projector → LLM
- LLM: Llama-2 7B (34 B params with 64× expansion in MLP)
- Pre-training: Open X-Embodiment ~970 K demonstrations (not BridgeData alone)
- Action space: 7-D continuous delta (dx, dy, dz, droll, dpitch, dyaw, gripper)
  normalised → [-1, 1] → discretised into 256 bins → Llama vocab token IDs

LoRA config
-----------
- ``target_modules="all-linear"`` (official OpenVLA finetune.py config)
- rank=32 → ≈ 463 MB trainable params (≈ 1.3 % of 34 B backbone)
- Vision backbone + projector **frozen**; only LM adapter trained

Sim-to-real gap
---------------
- Collected in NVIDIA Isaac Sim as **position-only DLS-IK expert trajectories**
- Continuous capture at 5–10 Hz during IK execution (not stage endpoints)
- Real G1-D transfer will require **delta-action conditioning** — the model
  learns (image, instruction) → next delta, not (image, instruction) → final pose
- Expect a **domain gap** between rendered cup/table textures and real-world
  lighting/backgrounds; consider fine-tuning with a small real-world set if
  available

GPU memory note
---------------
- Full OpenVLA-7B ≈ 15 GB in bf16; with LoRA adapters ≈ 15.5 GB
- RTX 4090 (24 GB): expects `batch_size=1–2` with gradient checkpointing
- For batch_size=1 on 4 GPUs with DataParallel: effective batch = 4

Usage (after collecting demos with --collect-grasp-demos)::

    torchrun --nproc_per_node=4 scripts/train_openvla_grasp.py \
        --demo-dir outputs/family_home_vln/grasp_demos \
        --model checkpoints/openvla-7b \
        --output checkpoints/openvla-g1d-grasp-lora \
        --epochs 10 --batch-size 1 --lr 5e-4

The script:
1. Loads all episodes from the grasp_demos directory.
2. Computes action normalisation statistics (q01/q99 per dim).
3. Splits into train / val (90 / 10 by default).
4. Applies LoRA to the LM backbone and fine-tunes.
5. Saves the LoRA adapter weights + ``dataset_statistics.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from PIL import Image
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForVision2Seq, AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IGNORE_INDEX = -100
ACTION_DIM = 7
ACTION_LABELS = ("dx_m", "dy_m", "dz_m", "droll_rad", "dpitch_rad", "dyaw_rad", "gripper")

OPENVLA_PROMPT = "In: What action should the robot take to {instruction}?\nOut:"

# Token 29871 is the special empty token that OpenVLA expects after ':'
SPECIAL_EMPTY_TOKEN = 29871


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GraspDemoDataset(Dataset):
    """PyTorch Dataset wrapping the grasp_demos directory layout.

    Each item yields::

        {
            "pixel_values": Tensor[3, H, W]          – RGB image as float32 [0,1],
            "input_ids":    LongTensor[seq_len]       – prompt + action tokens,
            "labels":       LongTensor[seq_len]       – IGNORE_INDEX for prompt,
            "action_7d":    FloatTensor[7]            – raw continuous action,
        }
    """

    def __init__(
        self,
        demo_dir: Path,
        processor: AutoProcessor,
        tokenizer,
        *,
        norm_stats: Optional[dict] = None,
        vocab_size: int = 32000,
        bin_centers: Optional[np.ndarray] = None,
    ) -> None:
        self.demo_dir = Path(demo_dir)
        self.processor = processor
        self.tokenizer = tokenizer
        self.norm_stats = norm_stats
        self.vocab_size = vocab_size
        self.bin_centers = bin_centers

        # Discover episodes
        self.episodes: list[Path] = sorted(
            p for p in self.demo_dir.glob("episode_*") if p.is_dir()
        )
        self.episodes = [
            episode
            for episode in self.episodes
            if self._episode_is_training_ready(episode)
        ]
        if not self.episodes:
            raise FileNotFoundError(
                f"No success-verified training episodes found in {self.demo_dir}"
            )

        # Build flat index: list of (episode_dir, step_index)
        self._steps: list[tuple[Path, int]] = []
        for ep in self.episodes:
            step_dirs = sorted(ep.glob("step_*"))
            for step_dir in step_dirs:
                if (step_dir / "image.png").is_file() and (step_dir / "action.json").is_file():
                    idx = int(step_dir.name.split("_")[-1])
                    self._steps.append((ep, idx))

        if not self._steps:
            raise RuntimeError(f"No valid (image.png + action.json) pairs in {self.demo_dir}")

    @staticmethod
    def _episode_is_training_ready(episode: Path) -> bool:
        """Accept only episodes that passed both RGB and physical gates."""

        meta_path = episode / "meta.json"
        if not meta_path.is_file():
            return False
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return bool(
            meta.get("success")
            and meta.get("ready_for_training")
            # Gate v2 explicitly measures large near-black regions in every
            # RGB observation.  Older episodes predate this check and contain
            # the black dining-table fascia, so they must not enter training.
            and int(meta.get("rgb_quality_gate_version", 0)) >= 2
            and int(meta.get("black_frame_count", 1)) == 0
        )

    def __len__(self) -> int:
        return len(self._steps)

    def _load_step(self, ep_dir: Path, step_idx: int) -> tuple[Image.Image, np.ndarray, str]:
        step_dir = ep_dir / f"step_{step_idx:04d}"
        image = Image.open(step_dir / "image.png").convert("RGB")
        action_data = json.loads((step_dir / "action.json").read_text(encoding="utf-8"))
        action = np.array(
            [action_data[label] for label in ACTION_LABELS], dtype=np.float32
        )
        meta = json.loads((ep_dir / "meta.json").read_text(encoding="utf-8"))
        instruction = meta.get("instruction", "move the robot hand toward the coffee cup")
        return image, action, instruction

    def load_raw_action(self, idx: int) -> np.ndarray:
        """Load one 7-D label without decoding/tokenising its RGB image."""

        ep_dir, step_idx = self._steps[idx]
        action_data = json.loads(
            (ep_dir / f"step_{step_idx:04d}" / "action.json").read_text(
                encoding="utf-8"
            )
        )
        return np.asarray(
            [action_data[label] for label in ACTION_LABELS], dtype=np.float32
        )

    def _normalise_action(self, action: np.ndarray) -> np.ndarray:
        """Normalise 7-D action to [-1, 1] using q01 / q99 stats."""
        if self.norm_stats is None:
            return np.clip(action, -1.0, 1.0).astype(np.float32)
        stats = self.norm_stats["action"]
        low = np.array(stats["q01"], dtype=np.float32)
        high = np.array(stats["q99"], dtype=np.float32)
        # Avoid division by zero for zero-range dims (e.g. rotation if disabled)
        span = high - low
        span[span < 1e-8] = 1.0
        return (2.0 * (action - low) / span - 1.0).astype(np.float32)

    def _action_to_token_ids(self, norm_action: np.ndarray) -> list[int]:
        """Discretise a normalised action into action token IDs."""
        if self.bin_centers is None:
            raise RuntimeError("bin_centers must be set to tokenise actions")
        ids = []
        for val in norm_action:
            val_c = np.clip(val, -1.0, 1.0)
            bin_id = int(np.argmin(np.abs(self.bin_centers - val_c)))
            token_id = self.vocab_size - 1 - bin_id
            ids.append(token_id)
        return ids

    def __getitem__(self, idx: int) -> dict:
        ep_dir, step_idx = self._steps[idx]
        image, action, instruction = self._load_step(ep_dir, step_idx)

        # Build prompt
        prompt = OPENVLA_PROMPT.format(instruction=instruction.strip().lower())

        # Tokenise prompt
        prompt_tokens = self.tokenizer(prompt, return_tensors="pt")
        prompt_ids = prompt_tokens.input_ids[0]  # LongTensor

        # Ensure special empty token after ':'
        if prompt_ids[-1].item() != SPECIAL_EMPTY_TOKEN:
            prompt_ids = torch.cat(
                [prompt_ids, torch.tensor([SPECIAL_EMPTY_TOKEN], dtype=torch.long)]
            )

        # Normalise + discretise action
        norm_action = self._normalise_action(action)
        action_token_ids = self._action_to_token_ids(norm_action)
        action_tensor = torch.tensor(action_token_ids, dtype=torch.long)

        # Build input_ids and labels
        input_ids = torch.cat([prompt_ids, action_tensor])
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        labels[-ACTION_DIM:] = action_tensor

        # Preprocess image into pixel_values
        # PrismaticProcessor requires paired text+image inputs.  Prompt/action
        # token IDs are assembled explicitly above, so call only its image
        # processor here to avoid tokenising the prompt a second time.
        processed = self.processor.image_processor(
            images=image, return_tensors="pt"
        )
        pixel_values = processed["pixel_values"][0]  # [3, H, W]

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "labels": labels,
            "action_7d": torch.from_numpy(action),
        }


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def collate_fn(batch: list[dict]) -> dict:
    """Pad input_ids / labels and stack pixel_values."""
    # Find max sequence length
    max_len = max(item["input_ids"].shape[0] for item in batch)

    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    input_ids_padded = torch.full((len(batch), max_len), fill_value=0, dtype=torch.long)
    labels_padded = torch.full((len(batch), max_len), fill_value=IGNORE_INDEX, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)

    for i, item in enumerate(batch):
        n = item["input_ids"].shape[0]
        input_ids_padded[i, :n] = item["input_ids"]
        labels_padded[i, :n] = item["labels"]
        attention_mask[i, :n] = True

    action_7d = torch.stack([item["action_7d"] for item in batch])

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids_padded,
        "labels": labels_padded,
        "attention_mask": attention_mask,
        "action_7d": action_7d,
    }


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------

def apply_lora_to_model(model: nn.Module, *, rank: int = 32, alpha: int = 64) -> nn.Module:
    """Apply LoRA to the language-model backbone, freezing vision + projector.

    Uses ``target_modules="all-linear"`` — the official OpenVLA LoRA config
    that attaches adapters to ALL linear layers in the LM backbone (Q, K, V, O,
    FFN up/gate/down, lm_head).  Rank-32 gives ≈ 463 MB of trainable params
    (≈ 1.3 % of the 34 B param LLM backbone).
    """
    from peft import LoraConfig, get_peft_model, TaskType

    # Freeze vision backbone (DINOv2 + SigLIP fused) and projector
    for param in model.vision_backbone.parameters():
        param.requires_grad = False
    for param in model.projector.parameters():
        param.requires_grad = False

    # Apply LoRA to the LM backbone
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.1,
        target_modules="all-linear",
        bias="none",
    )

    model.language_model = get_peft_model(model.language_model, lora_config)
    return model


# ---------------------------------------------------------------------------
# Action statistics
# ---------------------------------------------------------------------------

def compute_action_stats(all_actions: list[np.ndarray]) -> dict:
    """Compute q01 / q99 / mean / std for a 7-D action dataset."""
    data = np.stack(all_actions, axis=0)  # [N, 7]
    stats = {}
    for dim in range(data.shape[1]):
        col = data[:, dim]
        stats[dim] = {
            "q01": float(np.percentile(col, 1)),
            "q99": float(np.percentile(col, 99)),
            "mean": float(np.mean(col)),
            "std": float(np.std(col)),
            "min": float(np.min(col)),
            "max": float(np.max(col)),
        }
    return stats


def build_norm_stats(dataset: GraspDemoDataset, num_samples: int = 0) -> dict:
    """Aggregate action statistics across the dataset."""
    all_actions: list[np.ndarray] = []
    indices = range(len(dataset)) if num_samples <= 0 else range(min(num_samples, len(dataset)))
    for i in indices:
        all_actions.append(dataset.load_raw_action(i))

    per_dim = compute_action_stats(all_actions)
    n = len(all_actions)
    return {
        "g1d_cup_grasp": {
            "action": {
                "mask": [True, True, True, True, True, True, True],
                "q01": [per_dim[d]["q01"] for d in range(ACTION_DIM)],
                "q99": [per_dim[d]["q99"] for d in range(ACTION_DIM)],
                "mean": [per_dim[d]["mean"] for d in range(ACTION_DIM)],
                "std": [per_dim[d]["std"] for d in range(ACTION_DIM)],
                "min": [per_dim[d]["min"] for d in range(ACTION_DIM)],
                "max": [per_dim[d]["max"] for d in range(ACTION_DIM)],
            },
            "num_trajectories": 0,  # filled after loading
            "num_transitions": n,
        }
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch(
    model,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    max_grad_norm: float = 1.0,
    *,
    gradient_accumulation_steps: int = 1,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
) -> float:
    """Single training epoch with optional gradient accumulation.

    Returns average loss per *micro-batch* (not per parameter update).
    """
    model.train()
    model_dtype = model.module.dtype if hasattr(model, "module") else model.dtype
    total_loss = 0.0
    accum_counter = 0
    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device, dtype=model_dtype)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # Scale loss by accumulation steps so effective LR stays the same
        with torch.autocast(device_type="cuda", dtype=model_dtype):
            output = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = output.loss / gradient_accumulation_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        accum_counter += 1
        if accum_counter >= gradient_accumulation_steps:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()
            accum_counter = 0

        total_loss += loss.item() * gradient_accumulation_steps  # unscale for reporting

    # Do not discard the final partial accumulation window.
    if accum_counter:
        if scaler is not None:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()

    totals = torch.tensor(
        [total_loss, float(len(dataloader))],
        device=device,
        dtype=torch.float64,
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    return float(totals[0].item() / max(totals[1].item(), 1.0))


@torch.no_grad()
def validate_epoch(model, dataloader: DataLoader, device: torch.device) -> float:
    """Validation epoch. Returns average loss."""
    model.eval()
    model_dtype = model.module.dtype if hasattr(model, "module") else model.dtype
    total_loss = 0.0
    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device, dtype=model_dtype)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        output = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        total_loss += output.loss.item()

    totals = torch.tensor(
        [total_loss, float(len(dataloader))],
        device=device,
        dtype=torch.float64,
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    return float(totals[0].item() / max(totals[1].item(), 1.0))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--demo-dir", type=Path, required=True, help="grasp_demos directory")
    p.add_argument("--model", type=Path, required=True, help="base OpenVLA-7B checkpoint dir")
    p.add_argument("--output", type=Path, required=True, help="output directory for LoRA weights")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--gpus", type=int, default=1, metavar="N",
                   help="number of GPUs for DataParallel (default 1; 4 recommended)")
    p.add_argument("--gradient-accumulation-steps", type=int, default=1,
                   help="effective batch multiplier before weight update")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Resolve paths relative to project root
    for name in ("demo_dir", "model", "output"):
        path = getattr(args, name)
        if not path.is_absolute():
            setattr(args, name, (ROOT / path).resolve())

    if not args.demo_dir.is_dir():
        raise FileNotFoundError(f"--demo-dir not found: {args.demo_dir}")
    if not args.model.is_dir():
        raise FileNotFoundError(f"--model not found: {args.model}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(args.device)
    if not torch.cuda.is_available() and device.type == "cuda":
        raise RuntimeError("--device cuda requested but CUDA is unavailable")

    n_gpus = world_size if distributed else max(1, args.gpus)
    effective_batch = (
        args.batch_size * world_size * args.gradient_accumulation_steps
    )
    print(f"Rank: {rank}/{world_size}  device: {device}  available: {torch.cuda.device_count()}")
    print(f"Effective batch: {effective_batch}  (per-rank bs={args.batch_size} × ranks={world_size} × grad_accum={args.gradient_accumulation_steps})")
    print(f"Demo dir: {args.demo_dir}")
    print(f"Base model: {args.model}")
    print(f"Output: {args.output}")

    # ---- Load OpenVLA model + processor -----------------------------------
    print("\nLoading OpenVLA model ...")
    processor = AutoProcessor.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True
    )
    tokenizer = processor.tokenizer

    # Match OpenVLAForActionPrediction exactly: config.n_action_bins=256 is
    # used as the number of linspace boundary points, yielding 255 centres.
    # OpenVLA maps each normalised action dim → bin_id ∈ [0, 255] → token
    # token_id = vocab_size - 1 - bin_id  (last 256 tokens of LLM vocab)
    n_action_bins = 256
    bin_edges = np.linspace(-1, 1, n_action_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0  # shape (255,)
    vocab_size = 32000  # 32064 - 64 (pad_to_multiple_of)

    # ---- Build dataset ----------------------------------------------------
    print("Building dataset ...")
    # First pass: no norm_stats — collect raw actions for statistics
    raw_dataset = GraspDemoDataset(
        args.demo_dir,
        processor,
        tokenizer,
        norm_stats=None,
        vocab_size=vocab_size,
        bin_centers=bin_centers,
    )
    print(f"  Found {len(raw_dataset)} steps across {len(raw_dataset.episodes)} episodes")

    # Compute norm stats
    norm_stats = build_norm_stats(raw_dataset)
    stats_str = "  ".join(
        f"{lbl}: [{norm_stats['g1d_cup_grasp']['action']['q01'][d]:.4f}, "
        f"{norm_stats['g1d_cup_grasp']['action']['q99'][d]:.4f}]"
        for d, lbl in enumerate(ACTION_LABELS)
    )
    print(f"  Action q01/q99 ranges:\n    {stats_str}")

    # Second pass: norm_stats provided — actions will be properly normalised
    dataset = GraspDemoDataset(
        args.demo_dir,
        processor,
        tokenizer,
        norm_stats=norm_stats["g1d_cup_grasp"],
        vocab_size=vocab_size,
        bin_centers=bin_centers,
    )

    # Train / Val split
    if len(dataset) < 2:
        raise RuntimeError("At least two success-verified transitions are required")
    val_size = min(len(dataset) - 1, max(1, int(len(dataset) * args.val_split)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"  Train: {train_size}  Val: {val_size}")

    train_sampler = (
        DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
        )
        if distributed
        else None
    )
    val_sampler = (
        DistributedSampler(
            val_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )
        if distributed
        else None
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    # ---- Load model -------------------------------------------------------
    model = AutoModelForVision2Seq.from_pretrained(
        args.model,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=True,
    )
    # Inject our norm_stats so predict_action can use them
    model.norm_stats = norm_stats
    model.to(device)

    # Apply LoRA
    model = apply_lora_to_model(model, rank=args.lora_rank, alpha=args.lora_alpha)
    # A 7B VLA forward with the fused visual token sequence nearly fills a
    # 24-GB 4090 even at batch size one.  Checkpoint decoder activations and
    # disable inference-only KV caching before DataParallel replication.
    model.language_model.config.use_cache = False
    if hasattr(model.language_model, "enable_input_require_grads"):
        model.language_model.enable_input_require_grads()
    model.language_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # torchrun/DDP loads the custom fused visual backbone independently on
    # each device. DataParallel replication is not safe for its overridden
    # TIMM featurizer methods, which can retain cuda:0-bound state.
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank])
        print(f"  Rank {rank}: DistributedDataParallel on cuda:{local_rank}")
        _save_model = model.module
    elif n_gpus > 1:
        if n_gpus > torch.cuda.device_count():
            raise RuntimeError(
                f"Requested {n_gpus} GPUs but only {torch.cuda.device_count()} available"
            )
        gpu_ids = list(range(n_gpus))
        model = nn.DataParallel(model, device_ids=gpu_ids)
        print(f"  DataParallel across GPUs {gpu_ids}")
        _save_model = model.module
    else:
        _save_model = model

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # ---- Optimizer + scheduler --------------------------------------------
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    steps_per_epoch = math.ceil(
        len(train_loader) / args.gradient_accumulation_steps
    )
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=args.warmup_ratio,
    )

    # GradScaler is for float16 underflow protection.  This run uses BF16,
    # whose exponent range does not need scaling, and PyTorch cannot unscale
    # BF16 gradients with the CUDA foreach kernel.
    scaler = None

    # ---- Training loop ----------------------------------------------------
    print(f"\nTraining {args.epochs} epochs, {steps_per_epoch} steps/epoch"
          f"{' (pre-accumulation)' if args.gradient_accumulation_steps > 1 else ''} ...")
    best_val_loss = float("inf")
    best_epoch = 0
    training_history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_loss = train_epoch(
            model, train_loader, optimizer, device, scaler, args.max_grad_norm,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            scheduler=scheduler,
        )

        val_loss = validate_epoch(model, val_loader, device)

        lr_now = optimizer.param_groups[0]["lr"]
        if rank == 0:
            training_history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "learning_rate": lr_now,
                }
            )
            print(
                f"  Epoch {epoch:3d}/{args.epochs}  "
                f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
                f"lr={lr_now:.2e}"
            )

        if rank == 0 and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            args.output.mkdir(parents=True, exist_ok=True)
            # Unwrap DataParallel for saving
            lm = model.module.language_model if hasattr(model, "module") else model.language_model
            lm.save_pretrained(args.output)
            stats_path = args.output / "dataset_statistics.json"
            stats_path.write_text(
                json.dumps(norm_stats, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            tokenizer.save_pretrained(args.output)
            print(f"    Saved best checkpoint to {args.output}")

        if rank == 0:
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / "training_metrics.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "best_epoch": best_epoch,
                        "best_val_loss": best_val_loss,
                        "history": training_history,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

    if distributed:
        dist.barrier()
    if rank == 0:
        print(f"\nDone. Best val_loss: {best_val_loss:.6f}")
        print(f"LoRA adapter saved to: {args.output}")
    if distributed:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
