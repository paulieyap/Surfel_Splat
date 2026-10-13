#!/usr/bin/env python3
"""SurfelSplat depth for whole crop-monitor frames, written as a crop-monitor
dump so the team's own zed/make_cloud.py and recon/nn_score.py can fuse and
score it exactly like the stereo depth they replace.

    python custom/cm_scene.py --rgb RGB_DIR --colmap COLMAP_DUMP --out OUT \
        [--anchor-cloud FUSED_STEREO.ply] [--prior-depth 0.45]

RGB_DIR    left frames; names may be crop-monitor stems (000015.png) or the
           NAS 1 fps extraction (left_00001_svo0000015.png -> 000015)
COLMAP_DUMP a dump with intrinsics.json, frames.json, poses_colmap.txt

Each frame is covered by overlapping tiles with DTU's ~26 deg field of view
(the model's training distribution); each tile runs with the same tile in the
previous and next frame as extra context views, and the centre view's depth
is blended into the full-resolution depth map (Hann-weighted overlaps).

Two depth sets are written:
  depth_ss_raw   scene scaled once so --prior-depth maps to DTU distance; no
                 per-image information, i.e. SurfelSplat + poses alone
  depth_ss_anch  each tile rescaled by one factor: median ratio to the depth
                 of --anchor-cloud (an existing fused stereo cloud) rendered
                 into that view. Upper bound for SurfelSplat's shape when an
                 external source supplies the scale.
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np


def stem_of(name):
    m = re.search(r"svo(\d+)", name)
    return f"{int(m.group(1)):06d}" if m else Path(name).stem


def tiles(W, H, tw, th, overlap=0.25):
    def pos(L, t):
        n = max(1, int(np.ceil((L - t) / (t * (1 - overlap)))) + 1)
        return np.linspace(0, L - t, n).round().astype(int)
    return [(x, y) for y in pos(H, th) for x in pos(W, tw)]


def render_depth(K, T, pts, W, H, step=4):
    """z-buffer depth of a point cloud in a camera (T camera->world, CV)."""
    Xc = (pts - T[:3, 3]) @ T[:3, :3]
    z = Xc[:, 2]
    ok = z > 0.05
    u = (K["fx"] * Xc[ok, 0] / z[ok] + K["cx"]) / step
    v = (K["fy"] * Xc[ok, 1] / z[ok] + K["cy"]) / step
    z = z[ok]
    w, h = W // step, H // step
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v, z = u[ok].astype(int), v[ok].astype(int), z[ok]
    D = np.full(h * w, np.inf)
    np.minimum.at(D, v * w + u, z)
    D[np.isinf(D)] = 0
    return cv2.resize(D.reshape(h, w).astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb", type=Path, required=True)
    ap.add_argument("--colmap", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--anchor-cloud", type=Path)
    ap.add_argument("--prior-depth", type=float, default=0.45,
                    help="assumed typical camera distance (m) for the raw scale")
    ap.add_argument("--ref-depth-dir", type=Path,
                    help="optional stereo/MVS depth PNGs (mm) to score the depth maps against")
    ap.add_argument("--band", nargs=2, type=float, default=[0.15, 0.7])
    ap.add_argument("--cm-code", default="/workspace/data/ovais_code")
    args = ap.parse_args()

    sys.path.insert(0, args.cm_code)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from cropmon.poses import GL_TO_CV, load_tum
    from surfel_infer import IMAGE_HW, TARGET_DEPTH, load_encoder, run_views

    Kj = json.loads((args.colmap / "intrinsics.json").read_text())
    K = {k: float(Kj[k]) for k in ("fx", "fy", "cx", "cy")}
    W, H = int(Kj["width"]), int(Kj["height"])
    poses = load_tum(args.colmap / "poses_colmap.txt")
    frames_all = json.loads((args.colmap / "frames.json").read_text())
    rgb = {stem_of(p.name): p for p in args.rgb.glob("*.png")}
    frames = [f for f in frames_all if f["file"] in rgb and round(f["timestamp"], 6) in poses]
    T = {f["file"]: poses[round(f["timestamp"], 6)] @ GL_TO_CV for f in frames}
    print(f"{len(frames)} posed frames with rgb", flush=True)

    out = args.out
    for d in ("rgb", "depth_ss_raw", "depth_ss_anch", "conf_zero"):
        (out / d).mkdir(parents=True, exist_ok=True)
    shutil.copy(args.colmap / "intrinsics.json", out / "intrinsics.json")
    shutil.copy(args.colmap / "poses_colmap.txt", out / "poses_colmap.txt")
    (out / "frames.json").write_text(json.dumps(frames, indent=1))

    anchor = None
    if args.anchor_cloud:
        import open3d as o3d
        anchor = np.asarray(o3d.io.read_point_cloud(str(args.anchor_cloud)).points)
        print(f"anchor cloud {len(anchor)} pts", flush=True)

    enc = load_encoder()
    h, w = IMAGE_HW
    tw = int(round(min(K["fx"] / 2.18, W)))
    th = int(round(tw * h / w))
    boxes = tiles(W, H, tw, th)
    win = np.outer(np.hanning(th + 2)[1:-1], np.hanning(tw + 2)[1:-1]).astype(np.float32)
    k = TARGET_DEPTH / (1000 * args.prior_depth)            # metres -> model units
    print(f"tile {tw}x{th}, {len(boxes)} tiles/frame, scale {k * 1000:.1f} units/m", flush=True)

    imgs = {}

    def img(stem):
        if stem not in imgs:
            imgs[stem] = cv2.cvtColor(cv2.imread(str(rgb[stem])), cv2.COLOR_BGR2RGB)
        return imgs[stem]

    stats, t_inf = [], 0.0
    derr = {"raw": [], "anch": []}
    stems = [f["file"] for f in frames]
    for i, s in enumerate(stems):
        ctx = [stems[max(i - 1, 0)], s, stems[min(i + 1, len(stems) - 1)]]
        if ctx[0] == s:
            ctx[0] = stems[min(i + 2, len(stems) - 1)]
        if ctx[2] == s:
            ctx[2] = stems[max(i - 2, 0)]
        Tc = T[s]
        centre = Tc[:3, 3] + Tc[:3, 2] * args.prior_depth        # world origin near the scene
        acc_r = np.zeros((H, W), np.float32)
        acc_a = np.zeros((H, W), np.float32)
        wsum = np.zeros((H, W), np.float32)
        wsum_a = np.zeros((H, W), np.float32)
        D_anchor = render_depth(K, Tc, anchor, W, H) if anchor is not None else None
        for x0, y0 in boxes:
            ims, Ks, Ts = [], [], []
            for c in ctx:
                ims.append(cv2.resize(img(c)[y0:y0 + th, x0:x0 + tw], (w, h), interpolation=cv2.INTER_AREA))
                Ks.append(np.array([[K["fx"] / tw, 0, (K["cx"] - x0) / tw],
                                    [0, K["fy"] / th, (K["cy"] - y0) / th], [0, 0, 1]], np.float32))
                Tu = T[c].copy()
                Tu[:3, 3] = (Tu[:3, 3] - centre) * 1000 * k
                Ts.append(Tu.astype(np.float32))
            P, _, dt = run_views(enc, ims, Ks, Ts)
            t_inf += dt
            Pw = P[1] / (1000 * k) + centre
            z = ((Pw - Tc[:3, 3]) @ Tc[:3, :3])[..., 2]
            zt = cv2.resize(z.astype(np.float32), (tw, th), interpolation=cv2.INTER_LINEAR)
            sl = (slice(y0, y0 + th), slice(x0, x0 + tw))
            acc_r[sl] += zt * win
            wsum[sl] += win
            if D_anchor is not None:
                da = D_anchor[sl]
                ok = (da > 0.15) & (da < 0.7) & (zt > 0)
                if ok.sum() > 200:
                    a = float(np.median(da[ok] / zt[ok]))
                    acc_a[sl] += a * zt * win
                    wsum_a[sl] += win
                    stats.append(a)
        zr = np.where(wsum > 0, acc_r / np.maximum(wsum, 1e-6), 0)
        za = np.where(wsum_a > 0, acc_a / np.maximum(wsum_a, 1e-6), 0)
        cv2.imwrite(str(out / "depth_ss_raw" / f"{s}.png"), np.clip(zr * 1000, 0, 65535).astype(np.uint16))
        cv2.imwrite(str(out / "depth_ss_anch" / f"{s}.png"), np.clip(za * 1000, 0, 65535).astype(np.uint16))
        cv2.imwrite(str(out / "conf_zero" / f"{s}.png"), np.zeros((H, W), np.uint8))
        if args.ref_depth_dir and (args.ref_depth_dir / f"{s}.png").exists():
            zr_ref = cv2.imread(str(args.ref_depth_dir / f"{s}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000
            okr = (zr_ref > args.band[0]) & (zr_ref < args.band[1])
            for name, zz in (("raw", zr), ("anch", za)):
                ok = okr & (zz > 0)
                if ok.sum() > 1000:
                    e = np.abs(zz[ok] - zr_ref[ok]) * 1000
                    derr[name].append((float(np.median(e)), float(np.median(e / (zr_ref[ok] * 1000)))))
        dst = out / "rgb" / f"{s}.png"
        if not dst.exists():
            dst.symlink_to(rgb[s].resolve())
        imgs.pop(ctx[0], None)
        print(f"{i + 1}/{len(stems)} {s} raw median {np.median(zr[zr > 0]):.3f} m", flush=True)

    meta = {"frames": len(stems), "tiles_per_frame": len(boxes), "tile_px": [tw, th],
            "prior_depth_m": args.prior_depth, "infer_s_total": t_inf,
            "infer_s_per_frame": t_inf / max(len(stems), 1)}
    for name, v in derr.items():
        if v:
            meta[f"depth_abs_err_mm_median_{name}"] = float(np.median([a for a, _ in v]))
            meta[f"depth_rel_err_median_{name}"] = float(np.median([b for _, b in v]))
            meta[f"depth_err_frames_{name}"] = len(v)
    if stats:
        meta.update(anchor_scale_median=float(np.median(stats)),
                    anchor_scale_p10_p90=[float(np.percentile(stats, 10)), float(np.percentile(stats, 90))])
    (out / "ss_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
