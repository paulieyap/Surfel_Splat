#!/usr/bin/env python3
"""Convert packed crop-monitor ZED sessions (custom/train/nas_pack.py output)
into the repo's .torch training chunks.

    python custom/train/make_chunks.py RAW OUT --train S1 S2 ... --test S7

Each session is cut into windows of --window consecutive posed frames
(stride --stride), and each window into three 640x512 crops (left, centre,
right of the 910x512 frame). Every (window, crop) is one example.

Units: world and depth are metres * UNITS_PER_M (10/3), so the dataset's
near/far 0.5..4.5 cover 0.15..1.35 m. Depth PNGs hold units * 200 (the geo
loss divides by 200), i.e. 1.5 mm steps; 0 where FFS confidence > --conf-max
or outside the band. Normals are derived from that depth (only the loader
needs them; the recipe has no normal loss).
"""
import argparse
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

UNITS_PER_M = 10 / 3
CROPS = {"L": 0, "C": 136, "R": 270}          # x offsets in the 910x512 frame (even, so depth = /2)
CW, CH = 640, 512


def enc(img, ext, *params):
    return torch.from_numpy(np.frombuffer(cv2.imencode(ext, img, list(params))[1].tobytes(), np.uint8).copy())


def normals_from_depth(z, fx, fy, cx, cy):
    h, w = z.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    P = np.dstack([(u - cx) * z / fx, (v - cy) * z / fy, z])
    n = np.cross(np.gradient(P, axis=1), np.gradient(P, axis=0))
    n /= np.linalg.norm(n, axis=2, keepdims=True) + 1e-9
    n[z <= 0] = 0
    return ((n + 1) * 127.5).clip(0, 255).astype(np.uint8)


def session_examples(d, window, stride, conf_max, band, cm_code):
    sys.path.insert(0, cm_code)
    from cropmon.poses import GL_TO_CV, load_tum
    K = json.loads((d / "intrinsics.json").read_text())
    poses = load_tum(d / "poses_colmap.txt")
    frames = [f for f in json.loads((d / "frames.json").read_text())
              if round(f["timestamp"], 6) in poses and (d / "rgb" / f"{f['file']}.jpg").exists()
              and f.get("tracking", "OK") == "OK"]
    sx, sy = 910 / K["width"], 512 / K["height"]
    lo, hi = band
    out = []
    for w0 in range(0, max(len(frames) - window, 0) + 1, stride):
        win = frames[w0:w0 + window]
        if len(win) < 9:
            continue
        rgb = [cv2.imread(str(d / "rgb" / f"{f['file']}.jpg")) for f in win]
        dep = [cv2.imread(str(d / "depth" / f"{f['file']}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000
               for f in win]
        conf = [cv2.imread(str(d / "conf" / f"{f['file']}.png"), cv2.IMREAD_UNCHANGED) for f in win]
        w2c = []
        for f in win:
            T = poses[round(f["timestamp"], 6)] @ GL_TO_CV          # camera -> world, OpenCV axes, metres
            T = T.copy()
            T[:3, 3] *= UNITS_PER_M
            w2c.append(np.linalg.inv(T)[:3].reshape(-1))
        for name, x0 in CROPS.items():
            fx, fy = K["fx"] * sx, K["fy"] * sy
            cx, cy = K["cx"] * sx - x0, K["cy"] * sy
            cams, imgs, deps, nors = [], [], [], []
            for i in range(len(win)):
                imgs.append(enc(rgb[i][:, x0:x0 + CW], ".jpg", cv2.IMWRITE_JPEG_QUALITY, 95))
                z = dep[i][:, x0 // 2:x0 // 2 + CW // 2].copy()
                c = conf[i][:, x0 // 2:x0 // 2 + CW // 2] if conf[i] is not None else np.zeros_like(z, np.uint8)
                z[(c > conf_max) | (z < lo) | (z > hi)] = 0
                deps.append(enc(np.round(z * UNITS_PER_M * 200).astype(np.uint16), ".png"))
                nors.append(enc(cv2.cvtColor(normals_from_depth(z, fx / 2, fy / 2, cx / 2, cy / 2),
                                             cv2.COLOR_RGB2BGR), ".png"))
                cams.append(np.concatenate([[fx / CW, fy / CH, cx / CW, cy / CH, 0, 0], w2c[i]]))
            out.append({"key": f"{d.name}_{w0:04d}{name}", "url": "", "timestamps": torch.arange(len(win)),
                        "cameras": torch.tensor(np.stack(cams), dtype=torch.float32),
                        "images": imgs, "depths": deps, "normals": nors})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--test", nargs="+", default=[])
    ap.add_argument("--window", type=int, default=24)
    ap.add_argument("--stride", type=int, default=12)
    ap.add_argument("--conf-max", type=int, default=80)
    ap.add_argument("--band", nargs=2, type=float, default=[0.15, 1.35])
    ap.add_argument("--per-chunk", type=int, default=40)
    ap.add_argument("--cm-code", default="/workspace/data/cm_code")
    args = ap.parse_args()
    for stage, sessions in (("train", args.train), ("test", args.test)):
        (args.out / stage).mkdir(parents=True, exist_ok=True)
        exs = []
        for s in sessions:
            e = session_examples(args.raw / s, args.window, args.stride, args.conf_max, args.band, args.cm_code)
            print(f"{stage} {s}: {len(e)} examples", flush=True)
            exs += e
        if stage == "train":
            exs = [exs[i] for i in np.random.default_rng(0).permutation(len(exs))]
        index = {}
        for c in range(0, len(exs), args.per_chunk):
            name = f"{c // args.per_chunk:06d}.torch"
            torch.save(exs[c:c + args.per_chunk], args.out / stage / name)
            index.update({e["key"]: name for e in exs[c:c + args.per_chunk]})
        (args.out / stage / "index.json").write_text(json.dumps(index))
        print(f"{stage}: {len(exs)} examples in {(len(exs) + args.per_chunk - 1) // args.per_chunk} chunks", flush=True)


if __name__ == "__main__":
    main()
