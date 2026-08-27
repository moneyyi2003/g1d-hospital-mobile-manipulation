# G1-D OpenVLA-OFT v14 transfer bundle

This directory is self-contained for transfer to a training server.  It contains no symbolic links.

## Contents

- `training_data/`: 100 accepted v14 G1-D demonstrations, with RGB images, synchronized 7-D world-frame actions, and language instructions.
- `training_data/openvla_oft_manifest.json`: combined 100-episode manifest with statistics recomputed across both collection shards.
- `checkpoints/openvla-oft-libero-combined/`: the OpenVLA-OFT pretrained initialization checkpoint. Do not overwrite it.
- `third_party/openvla-oft/`: the OFT training source.
- `scripts/`: the G1-D manifest builder and OFT training launcher.

## Required environment

Create the OpenVLA-OFT Python environment on the destination server according to `third_party/openvla-oft/SETUP.md`. A CUDA GPU is required for training.

## Validate the transferred data

From this directory:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
  /path/to/openvla-oft/bin/python scripts/train_openvla_oft_g1d.py \
  --data-smoke \
  --demo-dir training_data \
  --manifest training_data/openvla_oft_manifest.json \
  --model checkpoints/openvla-oft-libero-combined
```

## Fine-tune

Run from this directory after the smoke test passes:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
  /path/to/openvla-oft/bin/python scripts/train_openvla_oft_g1d.py \
  --demo-dir training_data \
  --manifest training_data/openvla_oft_manifest.json \
  --model checkpoints/openvla-oft-libero-combined \
  --output checkpoints/openvla-oft-g1d-v14
```

The newly created `checkpoints/openvla-oft-g1d-v14/` is the directory to bring back after training. It includes the model checkpoint and G1-D action normalization metadata.
