#!/usr/bin/env python3
"""SurfelSplat on crop-monitor flower captures, scored with crop-monitor's own
flower code (flower/flower_fit.py at the snapshot in --cm-code).

    python custom/cm_flowers.py plan DUMP --tag ffs
    python custom/cm_flowers.py run  DUMP --tag ffs --depth-dir depth_ffs --out OUT

DUMP is a crop-monitor dump: rgb/, intrinsics.json, frames.json,
poses_colmap.txt (TUM camera->world, GL axes, metres), sam3/<cls>/*.json and
cloud3d_<tag>/objects_<cls>/flower_fits.json.

plan  per fitted flower, picks 3 context views from its member views (never
      its held-out silhouette test frames): the most facing view plus the
      pair whose viewing rays sit ~10 deg from it, like DTU neighbours.
      Writes OUT/plan.json with the frames whose rgb/depth must be fetched.
run   crops each context view around the flower, scales the scene so the
      flower sits at DTU distance, runs SurfelSplat, and compares, on the
      SAME three views, SurfelSplat's flower points with the dump's depth
      (FFS stereo on ZED, COLMAP MVS on GoPro):
        - depth error vs that depth on the flower mask, and coverage
        - view-to-view spread of the flower centre
        - flower_fit.fit_flower (plane + outline circle) on each point set
        - held-out IoU: each fitted 3D circle projected into the flower's
          sil_test_frames vs the convex hull of the best SAM3 mask
          (as draw_fits.py's iou_sil_hull), next to crop-monitor's fused
          all-view circle and its silhouette circle (Stage 7).
      Two crops: "tight" (flower ~1/3 of the crop) and "fov" (crop matches
      DTU's ~26 deg field of view).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

CROP_ASPECT = 320 / 256


def load_cm(cm_code):
    sys.path.insert(0, str(Path(cm_code) / "flower"))
    sys.path.insert(0, str(Path(cm_code)))
    import flower_fit as ff
    from cropmon.poses import GL_TO_CV, load_tum
    return ff, GL_TO_CV, load_tum


class Dump:
    def __init__(self, root, tag, cls, ff, GL_TO_CV, load_tum):
        self.root, self.cls, self.ff = Path(root), cls, ff
        k = json.loads((self.root / "intrinsics.json").read_text())
        self.K = {key: float(k[key]) for key in ("fx", "fy", "cx", "cy")}
        self.W, self.H = int(k["width"]), int(k["height"])
        poses = load_tum(self.root / "poses_colmap.txt")
        self.T = {}
        for fr in json.loads((self.root / "frames.json").read_text()):
            T = poses.get(round(fr["timestamp"], 6))
            if T is not None:
                self.T[fr["file"]] = T @ GL_TO_CV
        fits = json.loads((self.root / f"cloud3d_{tag}" / f"objects_{cls}" /
                           "flower_fits.json").read_text())
        self.fits = fits["flowers"]
        self.params = fits["params"]
        self._masks = {}

    def masks(self, stem):
        if stem not in self._masks:
            from pycocotools import mask as mask_utils
            jf = self.root / "sam3" / self.cls / f"{stem}.json"
            ann = json.loads(jf.read_text())["annotations"] if jf.exists() else []
            self._masks[stem] = [mask_utils.decode(a["segmentation"]).astype(bool)
                                 for a in ann]
        return self._masks[stem]

    def project(self, stem, X):
        return self.ff.project_px(self.K, self.T[stem], np.atleast_2d(X))


def flower_geom(f):
    c = np.array(f.get("sil_centre_mm") or f["centre_mm"]) / 1000
    n = np.array(f.get("sil_normal") or f["normal"], float)
    return c, n / np.linalg.norm(n)


def facing(T, c, n):
    ray = c - T[:3, 3]
    return float(abs(ray @ n) / np.linalg.norm(ray))


def flower_mask(d, stem, c, d_mm):
    """The SAM3 mask of this flower in frame stem: the one overlapping a disk
    of the flower's size at its projected centre the most."""
    (uv,), (z,) = d.project(stem, c)
    if z <= 0.02:
        return None, None
    r = d.K["fx"] * d_mm / 2000 / z
    yy, xx = np.ogrid[:d.H, :d.W]
    disk = (xx - uv[0]) ** 2 + (yy - uv[1]) ** 2 <= r * r
    best, score = None, 0.0
    for m in d.masks(stem):
        s = (m & disk).sum() / max(disk.sum(), 1)
        if s > score:
            best, score = m, s
    return (best if score > 0.25 else None), (uv, z, r)


def touches_border(m, px=3):
    return m[:px].any() or m[-px:].any() or m[:, :px].any() or m[:, -px:].any()


def ray_angle(Ta, Tb, c):
    a, b = c - Ta[:3, 3], c - Tb[:3, 3]
    return float(np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))


def plan(d, min_facing=0.7, target_deg=10.0):
    out = []
    for f in d.fits:
        d_mm = f.get("tip_d_mm") or 0
        test = [s for s in f.get("sil_test_frames") or [] if s in d.T]
        if d_mm <= 0 or len(test) < 2:
            continue
        c, n = flower_geom(f)
        cands = []
        for s in f["views"]:
            if s in test or s not in d.T:
                continue
            fc = facing(d.T[s], c, n)
            if fc < min_facing:
                continue
            m, geo = flower_mask(d, s, c, d_mm)
            if m is None or touches_border(m):
                continue
            uv, z, r = geo
            if not (2 * r <= uv[0] <= d.W - 2 * r and 2 * r <= uv[1] <= d.H - 2 * r):
                continue
            cands.append((s, fc))
        if len(cands) < 3:
            continue
        ref = max(cands, key=lambda x: x[1])[0]
        rest = [s for s, _ in cands if s != ref]
        best = None
        for i, a in enumerate(rest):
            for b in rest[i + 1:]:
                aa, ab = ray_angle(d.T[ref], d.T[a], c), ray_angle(d.T[ref], d.T[b], c)
                cost = abs(aa - target_deg) + abs(ab - target_deg) \
                    + max(0.0, target_deg - ray_angle(d.T[a], d.T[b], c))
                if best is None or cost < best[0]:
                    best = (cost, a, b)
        ctx = [ref, best[1], best[2]]
        out.append({
            "id": f["id"], "tip_d_mm": d_mm, "context": ctx, "test": test,
            "angles_deg": [round(ray_angle(d.T[ctx[0]], d.T[s], c), 1) for s in ctx[1:]],
            "facing": [round(facing(d.T[s], c, n), 3) for s in ctx],
        })
    return out


def crop_box(d, uv, z, d_mm, mode):
    if mode == "tight":
        cw = 3.0 * d.K["fx"] * d_mm / 1000 / z
    else:                                     # DTU fx/W = 2.18 -> ~26 deg
        cw = d.K["fx"] / 2.18
    cw = float(np.clip(cw, 96, d.W))
    ch = cw / CROP_ASPECT
    if ch > d.H:
        ch, cw = float(d.H), d.H * CROP_ASPECT
    cw, ch = int(round(cw)), int(round(ch))
    x0 = int(np.clip(round(uv[0] - cw / 2), 0, d.W - cw))
    y0 = int(np.clip(round(uv[1] - ch / 2), 0, d.H - ch))
    return x0, y0, cw, ch


def backproject(K, T, u, v, z):
    Xc = np.column_stack([(u - K["cx"]) / K["fx"] * z, (v - K["cy"]) / K["fy"] * z, z])
    return Xc @ T[:3, :3].T + T[:3, 3]


def voxel_thin(p, vox):
    if len(p) == 0:
        return p
    _, idx = np.unique(np.floor(p / vox).astype(np.int64), axis=0, return_index=True)
    return p[np.sort(idx)]


def recentre(views):
    """crop-monitor's step 1: shift each view's points so its median sits on
    the median of all views' medians. Returns points and per-view shift (mm)."""
    meds = [np.median(p, 0) for p in views]
    m = np.median(meds, 0)
    return np.vstack([p - md + m for p, md in zip(views, meds)]), \
        [float(np.linalg.norm(md - m) * 1000) for md in meds]


def held_out_iou(d, stem, c, n, r, n_face, c_face, min_facing=0.7):
    """draw_fits.py's iou_sil_hull for a 3D circle in one held-out frame. The
    frame is scored only if the flower faces it, judged by the flower's
    reference geometry (n_face, c_face), so every method is scored on the
    same frames whatever normal it fitted."""
    T = d.T[stem]
    if facing(T, c_face, n_face) < min_facing:
        return None
    uv, z = d.ff.project_px(d.K, T, d.ff.circle_points(c, n, r))
    if (z <= 0.02).any() or uv[:, 0].min() < 0 or uv[:, 1].min() < 0 \
            or uv[:, 0].max() >= d.W or uv[:, 1].max() >= d.H:
        return None
    masks = d.masks(stem)
    if not masks:
        return None
    pm = np.zeros((d.H, d.W), np.uint8)
    cv2.fillPoly(pm, [np.round(uv).astype(np.int32)], 1)
    pm = pm.astype(bool)
    k = int(np.argmax([(pm & m).sum() / max((pm | m).sum(), 1) for m in masks]))
    cnt = cv2.findContours(masks[k].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    hm = np.zeros((d.H, d.W), np.uint8)
    cv2.fillPoly(hm, [cv2.convexHull(np.vstack(cnt)).reshape(-1, 2)], 1)
    hm = hm.astype(bool)
    return float((pm & hm).sum() / max((pm | hm).sum(), 1))


def score_circle(d, test, c, n, r, ref):
    v = [held_out_iou(d, s, c, n, r, ref[1], ref[0]) for s in test]
    v = [x for x in v if x is not None]
    return (float(np.median(v)) if v else None), len(v)


def write_ply(path, pts, rgb):
    import open3d as o3d
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)
    pc.colors = o3d.utility.Vector3dVector(rgb / 255.0)
    o3d.io.write_point_cloud(str(path), pc)


def evaluate(d, pp, views_pts, test, ref):
    """crop-monitor's flower metrics for one set of per-view flower points."""
    out = {}
    views_pts = [p for p in views_pts if len(p) >= 10]
    if len(views_pts) < 2:
        return out
    pts, shifts = recentre(views_pts)
    out["view_shift_mm_max"] = max(shifts)
    ft = d.ff.fit_flower(pts, pp["plane_tol"], pp["slab_mm"] / 1000, pp["off_plane_mm"] / 1000)
    if not ft:
        return out
    c, n = np.array(ft["centre_mm"]) / 1000, np.array(ft["normal"])
    dist = (pts - c) @ n * 1000
    out.update({k: ft[k] for k in ("circle_d_mm", "plane_rms_mm", "off_plane_frac", "n_points")})
    out["thickness_mm_p10_p90"] = float(np.percentile(dist, 90) - np.percentile(dist, 10))
    out["heldout_iou"], out["n_heldout_scored"] = score_circle(d, test, c, n, ft["circle_d_mm"] / 2000, ref)
    out["_pts"] = pts
    return out


def run(d, args, flowers):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from surfel_infer import IMAGE_HW, TARGET_DEPTH, load_encoder, run_views

    enc = load_encoder()
    H, W = IMAGE_HW
    pp = d.params
    zmin, zmax = args.band
    by_id = {f["id"]: f for f in d.fits}
    results = []
    for fl in flowers:
        f = by_id[fl["id"]]
        c, n = flower_geom(f)
        ref = (c, n)                                   # facing test for every method
        c_tri = np.array(f.get("centre_tri_mm") or f.get("sil_centre_mm") or f["centre_mm"]) / 1000
        d_mm = fl["tip_d_mm"]
        fdir = Path(args.out) / f"flower_{fl['id']:02d}"
        fdir.mkdir(parents=True, exist_ok=True)
        rec = {"id": fl["id"], "tip_d_mm": d_mm, "context": fl["context"],
               "n_test": len(fl["test"]), "methods": {}}

        # crop-monitor's own circles (all member views, fused depth) and its Stage 7 silhouette
        cm = {"circle_d_mm": f["circle_d_mm"], "plane_rms_mm": f["plane_rms_mm"],
              "off_plane_frac": f["off_plane_frac"], "n_views": f["n_views"],
              "view_shift_mm_max": max(f.get("view_shift_mm") or [0])}
        cm["heldout_iou"], cm["n_heldout_scored"] = score_circle(
            d, fl["test"], np.array(f["centre_mm"]) / 1000, np.array(f["normal"]), f["circle_d_mm"] / 2000, ref)
        rec["methods"]["cm_fused_allviews"] = cm
        if f.get("sil_d_mm"):
            sil = {"circle_d_mm": f["sil_d_mm"]}
            sil["heldout_iou"], sil["n_heldout_scored"] = score_circle(
                d, fl["test"], np.array(f["sil_centre_mm"]) / 1000, np.array(f["sil_normal"]),
                f["sil_d_mm"] / 2000, ref)
            rec["methods"]["cm_silhouette"] = sil

        views = []
        for s in fl["context"]:
            m, (uv, z, r) = flower_mask(d, s, c, d_mm)
            img = cv2.cvtColor(cv2.imread(str(d.root / "rgb" / f"{s}.png")), cv2.COLOR_BGR2RGB)
            dep = cv2.imread(str(d.root / args.depth_dir / f"{s}.png"), cv2.IMREAD_UNCHANGED)
            dep = dep.astype(np.float32) / 1000
            dep[(dep < zmin) | (dep > zmax)] = 0                # crop-monitor's near band
            T = d.T[s]
            z_tri = float(((c_tri - T[:3, 3]) @ T[:3, :3])[2])
            views.append(dict(stem=s, mask=m, uv=uv, z=z, z_tri=z_tri, img=img, depth=dep))
        k = TARGET_DEPTH / (1000 * np.median([v["z"] for v in views]))   # metres -> model units

        # reference: the dump's own depth (FFS / MVS) on the same three views
        ref_views, cov = [], []
        for v in views:
            ys, xs = np.nonzero(v["mask"])
            zz = v["depth"][ys, xs]
            ok = zz > 0
            cov.append(float(ok.mean()))
            ref_views.append(voxel_thin(backproject(d.K, d.T[v["stem"]], xs[ok], ys[ok], zz[ok]), pp["voxel"]))
        r3 = evaluate(d, pp, ref_views, fl["test"], ref)
        r3["mask_coverage"] = float(np.mean(cov))
        if "_pts" in r3:
            write_ply(fdir / "ref_points.ply", r3["_pts"], np.tile([[60, 140, 230]], (len(r3["_pts"]), 1)))
        rec["methods"][f"ref_{args.depth_dir}_3views"] = r3

        gy, gx = np.mgrid[:H, :W]
        for mode in args.modes:
            imgs, Ks, Ts, boxes = [], [], [], []
            for v in views:
                x0, y0, cw, ch = crop_box(d, v["uv"], v["z"], d_mm, mode)
                crop = v["img"][y0:y0 + ch, x0:x0 + cw]
                interp = cv2.INTER_AREA if cw > W else cv2.INTER_CUBIC
                imgs.append(cv2.resize(crop, (W, H), interpolation=interp))
                Ks.append(np.array([[d.K["fx"] / cw, 0, (d.K["cx"] - x0) / cw],
                                    [0, d.K["fy"] / ch, (d.K["cy"] - y0) / ch],
                                    [0, 0, 1]], np.float32))
                T = d.T[v["stem"]].copy()
                T[:3, 3] = (T[:3, 3] - c) * 1000 * k           # flower centre at the origin
                Ts.append(T.astype(np.float32))
                boxes.append((x0, y0, cw, ch))
            P, opa, dt = run_views(enc, imgs, Ks, Ts)
            P = P / (1000 * k) + c                             # back to world metres
            raw, anch, derr, derr_crop, derr_anch, cols = [], [], [], [], [], []
            for vi, (v, (x0, y0, cw, ch)) in enumerate(zip(views, boxes)):
                ui = np.clip((x0 + (gx + 0.5) * cw / W).astype(int), 0, d.W - 1)
                vi_ = np.clip((y0 + (gy + 0.5) * ch / H).astype(int), 0, d.H - 1)
                msk = v["mask"][vi_, ui]
                T = d.T[v["stem"]]
                zss = ((P[vi] - T[:3, 3]) @ T[:3, :3])[..., 2]
                a = v["z_tri"] / np.median(zss[msk])           # per-view anchor (no stereo used)
                Pa = T[:3, 3] + a * (P[vi] - T[:3, 3])
                zr = v["depth"][vi_, ui]
                ok = zr > 0
                if (ok & msk).sum() >= 10:
                    derr.append(np.abs(zss - zr)[ok & msk] * 1000)
                    derr_anch.append(np.abs(a * zss - zr)[ok & msk] * 1000)
                if ok.sum() >= 100:
                    derr_crop.append(np.abs(zss - zr)[ok] * 1000)
                raw.append(voxel_thin(P[vi][msk], pp["voxel"]))
                anch.append(voxel_thin(Pa[msk], pp["voxel"]))
                cols.append((Pa.reshape(-1, 3), imgs[vi].reshape(-1, 3), msk.reshape(-1)))
            for name, pts_views, errs in ((f"surfelsplat_{mode}", raw, derr),
                                          (f"surfelsplat_{mode}_anchored", anch, derr_anch)):
                m = evaluate(d, pp, pts_views, fl["test"], ref)
                m.update(infer_s=dt, crop_px=[b[2] for b in boxes], mask_coverage=1.0)
                if errs:
                    m["depth_abs_err_mm_median_mask"] = float(np.median(np.concatenate(errs)))
                if derr_crop and not name.endswith("anchored"):
                    m["depth_abs_err_mm_median_crop"] = float(np.median(np.concatenate(derr_crop)))
                rec["methods"][name] = m
            for vi, im in enumerate(imgs):
                cv2.imwrite(str(fdir / f"{mode}_ctx{vi}.jpg"), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
            allp = np.concatenate([p for p, _, _ in cols])
            allc = np.concatenate([c_ for _, c_, _ in cols]).astype(np.float64)
            allm = np.concatenate([m_ for _, _, m_ in cols])
            write_ply(fdir / f"{mode}_anchored_surfels_all.ply", allp, allc)
            write_ply(fdir / f"{mode}_anchored_flower_points.ply", allp[allm], allc[allm])
        for m in rec["methods"].values():
            m.pop("_pts", None)
        results.append(rec)
        print(json.dumps({"id": rec["id"], "tip": round(d_mm, 1), **{
            name: [round(m.get("circle_d_mm", -1), 1), m.get("heldout_iou") and round(m["heldout_iou"], 2),
                   round(m.get("thickness_mm_p10_p90", -1), 1)]
            for name, m in rec["methods"].items()}}), flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "run"])
    ap.add_argument("dump", type=Path)
    ap.add_argument("--tag", default="ffs")
    ap.add_argument("--cls", default="yellow_flower")
    ap.add_argument("--depth-dir", default="depth_ffs")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--modes", nargs="+", default=["tight", "fov"])
    ap.add_argument("--cm-code", default="/workspace/data/cm_code")
    ap.add_argument("--band", nargs=2, type=float, default=[0.15, 1.2],
                    help="valid depth band in metres for the reference depth")
    args = ap.parse_args()
    ff, GL_TO_CV, load_tum = load_cm(args.cm_code)
    d = Dump(args.dump, args.tag, args.cls, ff, GL_TO_CV, load_tum)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.cmd == "plan":
        p = plan(d)
        frames = sorted({s for fl in p for s in fl["context"]})
        (args.out / "plan.json").write_text(json.dumps({"flowers": p, "frames": frames}, indent=1))
        for fl in p:
            print(fl["id"], fl["tip_d_mm"], fl["context"], fl["angles_deg"], fl["facing"], len(fl["test"]))
        print(f"{len(p)} flowers, {len(frames)} frames:", " ".join(frames))
    else:
        p = json.loads((args.out / "plan.json").read_text())["flowers"]
        t0 = time.time()
        res = run(d, args, p)
        (args.out / "results.json").write_text(json.dumps(
            {"dump": str(args.dump), "depth_ref": args.depth_dir, "cm_code_commit":
             (Path(args.cm_code) / "COMMIT").read_text().strip(), "wall_s": time.time() - t0,
             "flowers": res}, indent=1))


if __name__ == "__main__":
    main()
