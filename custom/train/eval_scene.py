#!/usr/bin/env python3
"""Depth from a crop-monitor pilot checkpoint (cost-volume branch, trained
by config/experiment/cm.yaml) for whole frames, written as a crop-monitor
dump so zed/make_cloud.py and recon/nn_score.py can fuse and score it.

    python custom/train/eval_scene.py --ckpt CKPT --rgb RGB_DIR --colmap DUMP --out OUT \
        [--ref-depth-dir DEPTH_PNG_MM --band 0.15 1.2] [--baseline 0.035]

Each frame is scaled to 512 px high and covered by 640-wide crops (left,
centre, right for 16:9), fed at 256x320 like training. Context views are the
previous and next frames whose camera is closest to --baseline metres away.
Crops are blended with a horizontal Hann window. Output depth_ss_cm/ (mm).
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

UNITS_PER_M = 10 / 3            # must match custom/train/make_chunks.py
NEAR, FAR = 0.5, 4.5            # config/experiment/cm.yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--rgb", type=Path, required=True)
    ap.add_argument("--colmap", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ref-depth-dir", type=Path)
    ap.add_argument("--band", nargs=2, type=float, default=[0.15, 1.2])
    ap.add_argument("--baseline", type=float, default=0.035)
    ap.add_argument("--cm-code", default="/workspace/data/cm_code")
    args = ap.parse_args()

    sys.path.insert(0, args.cm_code)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from cm_scene import stem_of
    from cropmon.poses import GL_TO_CV, load_tum
    from surfel_infer import load_encoder, run_views

    Kj = json.loads((args.colmap / "intrinsics.json").read_text())
    W, H = int(Kj["width"]), int(Kj["height"])
    poses = load_tum(args.colmap / "poses_colmap.txt")
    rgb = {stem_of(p.name): p for p in args.rgb.glob("*.png")}
    frames = [f for f in json.loads((args.colmap / "frames.json").read_text())
              if f["file"] in rgb and round(f["timestamp"], 6) in poses]
    T = {f["file"]: poses[round(f["timestamp"], 6)] @ GL_TO_CV for f in frames}
    stems = [f["file"] for f in frames]
    C = np.array([T[s][:3, 3] for s in stems])

    out = args.out
    for d in ("rgb", "depth_ss_cm", "conf_zero"):
        (out / d).mkdir(parents=True, exist_ok=True)
    shutil.copy(args.colmap / "intrinsics.json", out / "intrinsics.json")
    shutil.copy(args.colmap / "poses_colmap.txt", out / "poses_colmap.txt")
    (out / "frames.json").write_text(json.dumps(frames, indent=1))

    s = 512 / H
    Ws = int(round(W * s))
    xs = [0] if Ws <= 640 else sorted({0, (Ws - 640) // 2, Ws - 640})
    win = np.hanning(642)[1:-1].astype(np.float32)
    enc = load_encoder(args.ckpt, experiment="cm")

    def img(stem):
        return cv2.resize(cv2.cvtColor(cv2.imread(str(rgb[stem])), cv2.COLOR_BGR2RGB), (Ws, 512),
                          interpolation=cv2.INTER_AREA)

    derr, base_used, t_inf = [], [], 0.0
    for i, st in enumerate(stems):
        d = np.linalg.norm(C - C[i], axis=1)
        before = [j for j in range(len(stems)) if j < i] or [j for j in range(len(stems)) if j > i]
        after = [j for j in range(len(stems)) if j > i] or [j for j in range(len(stems)) if j < i]
        a = min(before, key=lambda j: abs(d[j] - args.baseline))
        b = min([j for j in after if j != a] or after, key=lambda j: abs(d[j] - args.baseline))
        base_used += [d[a], d[b]]
        ctx = [stems[a], st, stems[b]]
        ims = {c: img(c) for c in ctx}
        origin = T[st][:3, 3]
        acc = np.zeros((512, Ws), np.float32)
        wsum = np.zeros((512, Ws), np.float32)
        for x0 in xs:
            views, Ks, Ts = [], [], []
            for c in ctx:
                views.append(cv2.resize(ims[c][:, x0:x0 + 640], (320, 256), interpolation=cv2.INTER_AREA))
                Ks.append(np.array([[Kj["fx"] * s / 640, 0, (Kj["cx"] * s - x0) / 640],
                                    [0, Kj["fy"] * s / 512, Kj["cy"] * s / 512], [0, 0, 1]], np.float32))
                Tu = T[c].copy()
                Tu[:3, 3] = (Tu[:3, 3] - origin) * UNITS_PER_M
                Ts.append(Tu.astype(np.float32))
            P, _, dt = run_views(enc, views, Ks, Ts, near=NEAR, far=FAR)
            t_inf += dt
            Pw = P[1] / UNITS_PER_M + origin
            z = ((Pw - T[st][:3, 3]) @ T[st][:3, :3])[..., 2]
            zc = cv2.resize(z.astype(np.float32), (640, 512), interpolation=cv2.INTER_LINEAR)
            acc[:, x0:x0 + 640] += zc * win
            wsum[:, x0:x0 + 640] += win
        z = cv2.resize(acc / np.maximum(wsum, 1e-6), (W, H), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(str(out / "depth_ss_cm" / f"{st}.png"), np.clip(z * 1000, 0, 65535).astype(np.uint16))
        cv2.imwrite(str(out / "conf_zero" / f"{st}.png"), np.zeros((H, W), np.uint8))
        dst = out / "rgb" / f"{st}.png"
        if not dst.exists():
            dst.symlink_to(rgb[st].resolve())
        if args.ref_depth_dir and (args.ref_depth_dir / f"{st}.png").exists():
            zr = cv2.imread(str(args.ref_depth_dir / f"{st}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000
            ok = (zr > args.band[0]) & (zr < args.band[1]) & (z > 0)
            if ok.sum() > 1000:
                e = np.abs(z[ok] - zr[ok])
                derr.append((float(np.median(e)) * 1000, float(np.median(e / zr[ok]))))
        print(f"{i + 1}/{len(stems)} {st} median {np.median(z):.3f} m", flush=True)

    meta = {"ckpt": args.ckpt, "frames": len(stems), "crops_per_frame": len(xs),
            "baseline_m_median": float(np.median(base_used)), "infer_s_per_frame": t_inf / max(len(stems), 1)}
    if derr:
        meta.update(depth_abs_err_mm_median=float(np.median([a for a, _ in derr])),
                    depth_rel_err_median=float(np.median([b for _, b in derr])), depth_err_frames=len(derr))
    (out / "ss_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
