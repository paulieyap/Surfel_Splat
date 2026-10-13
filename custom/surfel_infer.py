"""Load the SurfelSplat checkpoint once and run the encoder on posed crops.

Used by the crop-monitor experiments in custom/. Run from the repo root inside
the surfelsplat container, e.g. `python custom/cm_flowers.py ...`.

The released checkpoint predicts each context view's depth with a monocular
DINOv2 head whose output is squashed into [near, far] = [1.5, 4.525] x 200
(300..905 "DTU mm"). Poses only place those depths in 3D. So a scene must be
scaled so the target sits ~650 units from the cameras, as DTU objects did.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from jaxtyping import install_import_hook

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

with install_import_hook(("src",), ("beartype", "beartype")):
    from src.config import load_typed_root_config
    from src.global_cfg import set_cfg
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder
    from src.model.model_wrapper import ModelWrapper

IMAGE_HW = (256, 320)        # fixed by the DINOv2 depth head
NEAR, FAR = 1.5, 4.525       # experiment/re10k.yaml, before the x200
TARGET_DEPTH = 650.0         # median DTU object distance, in model units


def load_encoder(checkpoint="checkpoints/checkpoint.ckpt", device="cuda"):
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(REPO / "config"), version_base=None):
        cfg_dict = compose(config_name="main", overrides=[
            "+experiment=re10k", f"checkpointing.load={checkpoint}"])
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)
    encoder, visualizer = get_encoder(cfg.model.encoder)
    decoder = get_decoder(cfg.model.decoder, cfg.dataset)
    wrapper = ModelWrapper.load_from_checkpoint(
        str(REPO / checkpoint), optimizer_cfg=cfg.optimizer, test_cfg=cfg.test,
        train_cfg=cfg.train, encoder=encoder, encoder_visualizer=visualizer,
        decoder=decoder, losses=[], step_tracker=None)
    wrapper.eval().to(device)
    return wrapper.encoder


@torch.no_grad()
def run_views(encoder, images, intrinsics, c2w_units, device="cuda"):
    """images: list of HxWx3 uint8 RGB at IMAGE_HW; intrinsics: list of 3x3
    normalised by the crop size; c2w_units: list of 4x4 OpenCV camera->world
    in model units (DTU-like mm). Returns (points [V,H,W,3] in model units,
    opacity [V,H,W], seconds)."""
    v = len(images)
    img = torch.from_numpy(np.stack(images)).float().div(255).permute(0, 3, 1, 2)
    ctx = {
        "image": img[None].to(device),
        "intrinsics": torch.from_numpy(np.stack(intrinsics)).float()[None].to(device),
        "extrinsics": torch.from_numpy(np.stack(c2w_units)).float()[None].to(device),
        "near": torch.full((1, v), NEAR, device=device),
        "far": torch.full((1, v), FAR, device=device),
    }
    torch.cuda.synchronize()
    t0 = time.time()
    g = encoder.forward(ctx, False, visualization_dump={})
    torch.cuda.synchronize()
    dt = time.time() - t0
    h, w = IMAGE_HW
    pts = rearrange(g.means[0], "(v h w s) c -> v h w s c", v=v, h=h, w=w)[:, :, :, 0]
    opa = rearrange(g.opacities[0], "(v h w s) -> v h w s", v=v, h=h, w=w)[..., 0]
    return pts.float().cpu().numpy(), opa.float().cpu().numpy(), dt
