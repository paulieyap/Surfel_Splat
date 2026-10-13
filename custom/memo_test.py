#!/usr/bin/env python3
"""Memorisation probe for the DINOv2 depth head: score context-view depth on
DTU-format test scenes for the original images and for mild perturbations
(horizontal mirror, -20% brightness, 8 px shift). A depth estimator that
generalises barely changes; one that recalls seen images degrades sharply.

    python custom/memo_test.py [scene ...]
"""
import sys
from pathlib import Path

import torch
from lightning_fabric.utilities.apply_func import apply_to_collection
from torch.utils.data import default_collate

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dtu_branch_test import dataset_cfg  # noqa: E402
from surfel_infer import load_encoder  # noqa: E402


def perturb(ctx, kind):
    c = dict(ctx)
    img = ctx["image"]
    if kind == "mirror":
        c["image"] = img.flip(-1)
    elif kind == "darker":
        c["image"] = img * 0.8
    elif kind == "shift8":
        c["image"] = torch.roll(img, shifts=8, dims=-1)
    return c


def unperturb(depth, kind):
    if kind == "mirror":
        return depth.flip(-1)
    if kind == "shift8":
        return torch.roll(depth, shifts=-8, dims=-1)
    return depth


@torch.no_grad()
def main():
    from src.dataset import get_dataset
    from src.dataset.view_sampler.view_sampler_arbitrary import ViewSamplerArbitraryCfg
    scenes = sys.argv[1:] or ["scan24_train", "scan37_train", "scan63_train", "scan110_train", "Dog", "Stone"]
    cfg = dataset_cfg()
    enc = load_encoder()
    kinds = ("original", "mirror", "darker", "shift8")
    res = {k: [] for k in kinds}
    for s in scenes:
        cfg.dataset.view_sampler = ViewSamplerArbitraryCfg("arbitrary", 3, 1, context_views=[0, 2, 9], target_views=[4])
        cfg.dataset.single_scene = s
        ex = apply_to_collection(default_collate([next(iter(get_dataset(cfg.dataset, "test", None)))]),
                                 torch.Tensor, lambda x: x.cuda())
        gt = ex["context"]["depth"][0, :, 0].float()
        ok = (gt > 100) & (gt < 905)
        ok[..., :8] = False
        ok[..., -8:] = False
        line = [s]
        for k in kinds:
            enc.forward(perturb(ex["context"], k), False, visualization_dump=(vd := {}))
            d = unperturb(vd["depth"][0, ..., 0, 0].float(), k)
            e = (d - gt).abs()[ok]
            res[k].append(float(e.median()))
            line.append(f"{k} {e.median():6.1f} mm")
        print("  ".join(line), flush=True)
    print("median over scenes: " + "  ".join(f"{k} {torch.tensor(v).median():.1f} mm" for k, v in res.items()))


if __name__ == "__main__":
    main()
