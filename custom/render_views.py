#!/usr/bin/env python3
"""Render point clouds from a dump frame's camera (z-buffered splats), and
colour-mapped depth maps, for side-by-side figures.

    python custom/render_views.py clouds DUMP STEM OUT.jpg cloud1.ply:label [...]
    python custom/render_views.py depths DUMP STEM OUT.jpg depthdir1:label [...]
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def camera(dump, stem, cm_code="/workspace/data/cm_code"):
    sys.path.insert(0, cm_code)
    from cropmon.poses import GL_TO_CV, load_tum
    K = json.loads((Path(dump) / "intrinsics.json").read_text())
    P = load_tum(Path(dump) / "poses_colmap.txt")
    ts = {f["file"]: round(f["timestamp"], 6) for f in json.loads((Path(dump) / "frames.json").read_text())}
    return K, P[ts[stem]] @ GL_TO_CV


def splat(K, T, pts, cols, scale=0.5, radius=1):
    W, H = int(K["width"] * scale), int(K["height"] * scale)
    Xc = (pts - T[:3, 3]) @ T[:3, :3]
    ok = Xc[:, 2] > 0.05
    Xc, cols = Xc[ok], cols[ok]
    u = (K["fx"] * Xc[:, 0] / Xc[:, 2] + K["cx"]) * scale
    v = (K["fy"] * Xc[:, 1] / Xc[:, 2] + K["cy"]) * scale
    order = np.argsort(-Xc[:, 2])                    # far first, near overwrite
    img = np.full((H, W, 3), 24, np.uint8)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            uu = np.round(u[order]).astype(int) + dx
            vv = np.round(v[order]).astype(int) + dy
            m = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
            img[vv[m], uu[m]] = cols[order][m]
    return img


def label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (min(out.shape[1], 14 + 11 * len(text)), 30), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main():
    mode, dump, stem, out = sys.argv[1:5]
    K, T = camera(dump, stem)
    panels = []
    rgb = cv2.imread(str(Path(dump) / "rgb" / f"{stem}.png"))
    panels.append(label(cv2.resize(rgb, None, fx=0.5, fy=0.5), "photo"))
    for spec in sys.argv[5:]:
        path, name = spec.rsplit(":", 1)
        if mode == "clouds":
            import open3d as o3d
            pc = o3d.io.read_point_cloud(path)
            pts = np.asarray(pc.points)
            cols = (np.asarray(pc.colors)[:, ::-1] * 255).astype(np.uint8) if pc.has_colors() \
                else np.full((len(pts), 3), 200, np.uint8)
            panels.append(label(splat(K, T, pts, cols), name))
        else:
            d = cv2.imread(str(Path(path) / f"{stem}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000
            valid = d > 0
            lo, hi = 0.15, 1.0
            vis = cv2.applyColorMap((np.clip((d - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8),
                                    cv2.COLORMAP_TURBO)
            vis[~valid] = 24
            panels.append(label(cv2.resize(vis, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST), name))
    h = min(p.shape[0] for p in panels)
    panels = [cv2.resize(p, (int(p.shape[1] * h / p.shape[0]), h)) for p in panels]
    cv2.imwrite(out, np.hstack(panels), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("wrote", out)


if __name__ == "__main__":
    main()
