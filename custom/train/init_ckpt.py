#!/usr/bin/env python3
"""Initial weights for the crop-monitor pilot: the released checkpoint minus
the DINOv2 depth branch, and minus any tensor whose shape changed (the cost
volume layers sized by num_depth_candidates). Lightning's strict=False still
fails on shape mismatches, so they are dropped here.

    python custom/train/init_ckpt.py OUT.ckpt [hydra overrides, e.g. +experiment=cm]
"""
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def main():
    from hydra import compose, initialize_config_dir
    from src.config import load_typed_root_config
    from src.global_cfg import set_cfg
    from src.model.encoder import get_encoder
    out, overrides = sys.argv[1], sys.argv[2:] or ["+experiment=cm"]
    with initialize_config_dir(config_dir=str(REPO / "config"), version_base=None):
        cfg_dict = compose(config_name="main", overrides=overrides)
    set_cfg(cfg_dict)
    enc, _ = get_encoder(load_typed_root_config(cfg_dict).model.encoder)
    own = {f"encoder.{k}": v.shape for k, v in enc.state_dict().items()}
    ck = torch.load(REPO / "checkpoints/checkpoint.ckpt", map_location="cpu")
    keep, dropped = {}, []
    for k, v in ck["state_dict"].items():
        if k in own and own[k] == v.shape:
            keep[k] = v
        else:
            dropped.append(k)
    missing = [k for k in own if k not in keep]
    print(f"kept {len(keep)} tensors ({sum(v.numel() for v in keep.values()) / 1e6:.1f}M), "
          f"dropped {len(dropped)}, freshly initialised {len(missing)}: "
          f"{sorted({'.'.join(k.split('.')[:3]) for k in missing})}")
    torch.save({"state_dict": keep, "pytorch-lightning_version": ck["pytorch-lightning_version"]}, out)


if __name__ == "__main__":
    main()
