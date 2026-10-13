#!/usr/bin/env python3
"""Edge-on views of reconstructed flower points: the reference stereo/MVS
points and SurfelSplat (fov / tight, anchored) for a few flowers. The frame is
the PCA plane of the reference points, so a flat flower is a thin line.

    python custom/fig_flowers.py OUT.png EXPDIR:ID [...]
"""
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def load(p):
    return np.asarray(o3d.io.read_point_cloud(str(p)).points) if p.exists() else np.zeros((0, 3))


def main():
    out, specs = sys.argv[1], sys.argv[2:]
    sets = [("ref_points.ply", "stereo / MVS (3 views)", "#3c8ce6"),
            ("fov_anchored_flower_points.ply", "SurfelSplat fov, anchored", "#e07a2e"),
            ("tight_anchored_flower_points.ply", "SurfelSplat tight, anchored", "#c23b6b")]
    fig, axes = plt.subplots(len(specs), 1 + len(sets), figsize=(3.2 * (1 + len(sets)), 3.0 * len(specs)))
    axes = np.atleast_2d(axes)
    for row, spec in enumerate(specs):
        exp, fid = spec.rsplit(":", 1)
        fdir = Path(exp) / f"flower_{int(fid):02d}"
        rec = next(r for r in json.loads((Path(exp) / "results.json").read_text())["flowers"]
                   if r["id"] == int(fid))
        tip = rec["tip_d_mm"]
        ref = load(fdir / "ref_points.ply")
        if len(ref) < 50:                       # no stereo/MVS on the flower: frame from SurfelSplat
            ref = load(fdir / "fov_anchored_flower_points.ply")
        c = np.median(ref, 0)
        _, _, Vt = np.linalg.svd(ref - c, full_matrices=False)
        img = cv2.cvtColor(cv2.imread(str(fdir / "fov_ctx1.jpg")), cv2.COLOR_BGR2RGB)
        axes[row, 0].imshow(img)
        axes[row, 0].set_title(f"{Path(exp).name.replace('flowers_', '').upper()} flower {fid}  (tip_d {tip:.0f} mm)",
                               fontsize=9)
        axes[row, 0].axis("off")
        for col, (fn, name, colr) in enumerate(sets, 1):
            p = load(fdir / fn)
            ax = axes[row, col]
            if len(p):
                q = (p - c) @ Vt.T * 1000
                if len(q) > 6000:
                    q = q[np.random.default_rng(0).choice(len(q), 6000, replace=False)]
                ax.scatter(q[:, 0], q[:, 2], s=1, c=colr, alpha=0.5, linewidths=0)
            ax.plot([-tip / 2, tip / 2], [0, 0], color="k", lw=1.5)
            ax.set_xlim(-60, 60)
            ax.set_ylim(-60, 60)
            ax.set_aspect("equal")
            ax.grid(alpha=0.25)
            ax.tick_params(labelsize=7)
            if row == 0:
                ax.set_title(name, fontsize=9)
            if col == 1:
                ax.set_ylabel("off-plane, mm", fontsize=8)
            if row == len(specs) - 1:
                ax.set_xlabel("in-plane, mm (black = tip_d)", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print("wrote", out)


if __name__ == "__main__":
    main()
