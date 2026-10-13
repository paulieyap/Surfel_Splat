#!/usr/bin/env python3
"""Run ON THE NAS (python3 - < nas_pack.py ROOT SESSION ...): downscale each
session's posed frames and stream them as a tar on stdout, so only small
files cross the slow link. Writes nothing on the NAS.

Per frame: rgb/STEM.jpg 910x512 (q95), depth/STEM.png 455x256 uint16 mm
(nearest), conf/STEM.png 455x256 (nearest). Plus intrinsics.json,
frames.json and the pose file, unchanged.
"""
import io
import sys
import tarfile
import time
from pathlib import Path

import cv2

IMG_WH, DEP_WH = (910, 512), (455, 256)
root, sessions = Path(sys.argv[1]), sys.argv[2:]
tar = tarfile.open(fileobj=sys.stdout.buffer, mode="w|")


def add(name, data):
    ti = tarfile.TarInfo(name)
    ti.size, ti.mtime = len(data), int(time.time())
    tar.addfile(ti, io.BytesIO(data))


for s in sessions:
    d = root / s
    for f in ("intrinsics.json", "frames.json", "poses_colmap.txt"):
        add(f"{s}/{f}", (d / f).read_bytes())
    stems = sorted(p.stem for p in (d / "rgb").glob("*.png"))
    for i, stem in enumerate(stems):
        rgb = cv2.imread(str(d / "rgb" / f"{stem}.png"))
        dep = cv2.imread(str(d / "depth_ffs" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
        if rgb is None or dep is None:
            continue
        add(f"{s}/rgb/{stem}.jpg", cv2.imencode(".jpg", cv2.resize(rgb, IMG_WH, interpolation=cv2.INTER_AREA),
                                                [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes())
        add(f"{s}/depth/{stem}.png", cv2.imencode(".png", cv2.resize(dep, DEP_WH, interpolation=cv2.INTER_NEAREST))[1].tobytes())
        conf = cv2.imread(str(d / "conf_ffs" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
        if conf is not None:
            add(f"{s}/conf/{stem}.png", cv2.imencode(".png", cv2.resize(conf, DEP_WH, interpolation=cv2.INTER_NEAREST))[1].tobytes())
        if i % 50 == 0:
            print(f"{s} {i}/{len(stems)}", file=sys.stderr, flush=True)
tar.close()
