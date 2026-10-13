#!/usr/bin/env python3
"""Is the checkpoint's multi-view cost-volume depth branch trained? Run both
depth branches on DTU scenes (the model's own domain) and score the context
views' depth against DTU ground truth.

    python custom/dtu_branch_test.py [scan ...]   # numbers -> scanN_train; other names used as-is
"""
import json
import sys
from pathlib import Path

import torch
from lightning_fabric.utilities.apply_func import apply_to_collection
from torch.utils.data import default_collate

sys.path.insert(0, str(Path(__file__).resolve().parent))
from surfel_infer import REPO, load_encoder  # noqa: E402


def dataset_cfg():
    from hydra import compose, initialize_config_dir
    from src.config import load_typed_root_config
    with initialize_config_dir(config_dir=str(REPO / "config"), version_base=None):
        cfg_dict = compose(config_name="main", overrides=["+experiment=re10k"])
    return load_typed_root_config(cfg_dict)


@torch.no_grad()
def main():
    from src.dataset import get_dataset
    from src.dataset.view_sampler.view_sampler_arbitrary import ViewSamplerArbitraryCfg
    scans = sys.argv[1:] or ["24", "37", "40", "55", "63", "65", "69", "83", "97", "105", "106", "110", "114", "118", "122"]
    cfg = dataset_cfg()
    enc = load_encoder()
    rows = []
    for scan in scans:
        for ctx in ([0, 2, 9], [1, 4, 7], [3, 5, 8]):
            cfg.dataset.view_sampler = ViewSamplerArbitraryCfg("arbitrary", 3, 1, context_views=ctx, target_views=[4])
            cfg.dataset.single_scene = f"scan{scan}_train" if scan.isdigit() else scan
            try:
                ex = default_collate([next(iter(get_dataset(cfg.dataset, "test", None)))])
            except (StopIteration, ValueError, KeyError) as e:
                print(f"scan{scan}: not in test split ({e!r})", flush=True)
                break
            ex = apply_to_collection(ex, torch.Tensor, lambda x: x.cuda())
            gt = ex["context"]["depth"][0, :, 0].float()                     # v h w, DTU mm
            ok = (gt > 100) & (gt < 905)
            row = {"scan": scan, "ctx": ctx}
            for name, flag in (("dinov2", True), ("costvolume", False)):
                enc.depth_predictor.use_dinov2 = flag
                g = enc.forward(ex["context"], False, visualization_dump=(vd := {}))
                d = vd["depth"][0, ..., 0, 0].float()                         # v h w
                e = (d - gt).abs()[ok]
                row[name] = {"abs_mm": float(e.median()), "rel": float((e / gt[ok]).median()),
                             "pred_med": float(d[ok].median())}
            row["gt_med"] = float(gt[ok].median())
            rows.append(row)
            print(json.dumps(row), flush=True)
    for name in ("dinov2", "costvolume"):
        a = torch.tensor([r[name]["abs_mm"] for r in rows])
        r_ = torch.tensor([r[name]["rel"] for r in rows])
        print(f"{name:11s} median abs {a.median():.1f} mm  rel {r_.median():.3f}  over {len(rows)} runs")


if __name__ == "__main__":
    main()
