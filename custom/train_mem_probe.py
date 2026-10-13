#!/usr/bin/env python3
"""Peak GPU memory and step time of real training steps (encoder, renderer,
the experiment's losses, Adam over trainable params), for planning a
crop-monitor training run.

    [EFF=1] [CKPT=1] [BF16=1] python custom/train_mem_probe.py [dinov2 H W batch n_target [depth_candidates]]

EFF=1 swaps the surfel feature aggregation's unfold attention for an
equivalent windowed loop (checked numerically); CKPT=1 checkpoints that block; BF16=1 runs the encoder
under bf16 autocast; TRACE=1 prints memory around the encoder's blocks.
"""
import os
import sys
import time
from pathlib import Path

import torch
from lightning_fabric.utilities.apply_func import apply_to_collection
from torch.utils.data import default_collate

sys.path.insert(0, str(Path(__file__).resolve().parent))
from surfel_infer import REPO  # noqa: E402


def windowed_attention(fa, q, k, v, window_sizes, max_ws=5):
    """Same result as the repo's F.unfold version (zero padding, row-major
    window offsets, dynamic window mask, softmax over the window), but loops
    over the 25 offsets so it never materialises [B, C, 25, H, W]."""
    import torch.nn.functional as F
    b, _, h, w = q.shape
    p = max_ws // 2
    kp, vp = F.pad(k, (p, p, p, p)), F.pad(v, (p, p, p, p))
    offs = [(dy, dx) for dy in range(max_ws) for dx in range(max_ws)]
    scores = torch.stack([(q * kp[:, :, dy:dy + h, dx:dx + w]).sum(1) for dy, dx in offs], 1) * fa.scale
    mask = fa._create_window_mask(window_sizes, max_ws, h, w, b, q.device)[:, 0]     # [B, 25, H, W]
    attn = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=1)
    out = 0
    for i, (dy, dx) in enumerate(offs):
        out = out + attn[:, i:i + 1] * vp[:, :, dy:dy + h, dx:dx + w]
    return out


def efficient_feature_aggregation(enc):
    """Swap the two unfold-based local attentions for windowed_attention."""
    import types
    fa = enc.feature_aggregation

    def self_attn(self, x, window_sizes):
        xn = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        out = windowed_attention(self, self.normal_query_proj(xn), self.normal_key_proj(xn),
                                 self.normal_value_proj(xn), window_sizes)
        return self.normal_output_proj(out) + x

    def cross_attn(self, img_feats, refined_normal_feats, window_sizes):
        imn = self.norm2(img_feats.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        nn_ = self.norm2(refined_normal_feats.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        out = windowed_attention(self, self.img_proj(imn), self.refined_normal_k_proj(nn_),
                                 self.refined_normal_v_proj(nn_), window_sizes)
        return self.cross_output_proj(out) + img_feats

    fa._local_attention_for_normal_dynamic = types.MethodType(self_attn, fa)
    fa._local_cross_attention_dynamic = types.MethodType(cross_attn, fa)


@torch.no_grad()
def check_equivalence(enc):
    """Original vs windowed attention on random features of a small map."""
    fa = enc.feature_aggregation
    c = fa.feature_dim
    x = torch.randn(2, c, 24, 30, device="cuda")
    y = torch.randn(2, c, 24, 30, device="cuda")
    ws = torch.randint(1, 3, (2, 24, 30), device="cuda") * 2 + 1          # windows 3 or 5
    a = (fa._local_attention_for_normal_dynamic(x, ws), fa._local_cross_attention_dynamic(x, y, ws))
    efficient_feature_aggregation(enc)
    b = (fa._local_attention_for_normal_dynamic(x, ws), fa._local_cross_attention_dynamic(x, y, ws))
    return max(float((u - v).abs().max()) for u, v in zip(a, b))


def checkpoint_feature_aggregation(enc):
    """Recompute the full-resolution 5x5 local attention during backward
    instead of storing its unfolded tensors (~4.5 GiB each at batch 1)."""
    from torch.utils.checkpoint import checkpoint
    fa = enc.feature_aggregation
    fwd = fa.forward
    fa.forward = lambda *a, **k: checkpoint(fwd, *a, use_reentrant=False, **k)


def build(dinov2, image_shape, n_cand=None):
    from hydra import compose, initialize_config_dir
    from src.config import load_typed_root_config
    from src.global_cfg import set_cfg
    from src.loss import get_losses
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder
    with initialize_config_dir(config_dir=str(REPO / "config"), version_base=None):
        cfg_dict = compose(config_name="main", overrides=[
            "+experiment=re10k", "dataset.view_sampler.num_context_views=3",
            f"dataset.image_shape=[{image_shape[0]},{image_shape[1]}]",
            *([f"model.encoder.num_depth_candidates={n_cand}"] if n_cand else [])])
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)
    enc, _ = get_encoder(cfg.model.encoder)
    dec = get_decoder(cfg.model.decoder, cfg.dataset)
    sd = torch.load(REPO / "checkpoints/checkpoint.ckpt", map_location="cpu")["state_dict"]
    own = enc.state_dict()                # layers whose shape depends on the candidate count start fresh
    enc.load_state_dict({k[8:]: v for k, v in sd.items()
                         if k.startswith("encoder.") and own.get(k[8:], v).shape == v.shape}, strict=False)
    enc.depth_predictor.use_dinov2 = dinov2
    if not dinov2:                       # unused in the cost-volume branch: free ~1.7 GB
        enc.depth_predictor.dinov2_backbone = None
        enc.depth_predictor.monocular_depth_head = None
    return cfg, enc.cuda().train(), dec.cuda(), [l.cuda() for l in get_losses(cfg.loss)]


def example(cfg, b, n_target):
    from src.dataset import get_dataset
    from src.dataset.view_sampler.view_sampler_arbitrary import ViewSamplerArbitraryCfg
    cfg.dataset.view_sampler = ViewSamplerArbitraryCfg(
        "arbitrary", 3, n_target, context_views=[0, 2, 9], target_views=[4, 5, 6, 7][:n_target])
    cfg.dataset.single_scene = "scan24_train"
    ex = next(iter(get_dataset(cfg.dataset, "test", None)))
    return apply_to_collection(default_collate([ex] * b), torch.Tensor, lambda x: x.cuda())


def step(enc, dec, losses, opt, batch, bf16=False):
    _, _, _, h, w = batch["target"]["image"].shape
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
        g = enc(batch["context"], 0, False, scene_names=batch["scene"])
    g = apply_to_collection(g, torch.Tensor, lambda x: x.float())
    out = dec.forward(g, batch["target"]["extrinsics"], batch["target"]["intrinsics"],
                      batch["target"]["near"], batch["target"]["far"], (h, w), depth_mode="depth")
    loss = sum(l.forward(out, batch, g, 0) for l in losses)
    fwd = torch.cuda.max_memory_allocated() / 2**30
    print(f"  forward peak {fwd:.1f} GiB", flush=True)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()


def main():
    n_cand = None
    if len(sys.argv) > 1:                # one run: dinov2(0/1) H W batch n_target [depth candidates]
        a = sys.argv[1:]
        runs = [(a[0] == "1", (int(a[1]), int(a[2])), int(a[3]), int(a[4]))]
        n_cand = int(a[5]) if len(a) > 5 else None
    else:
        runs = [(False, (256, 320), 1, 1), (False, (256, 320), 2, 1), (False, (256, 320), 4, 1),
            (False, (256, 320), 2, 2), (False, (384, 512), 1, 1), (False, (384, 512), 2, 1),
            (True, (256, 320), 1, 1), (True, (256, 320), 2, 1)]
    for dinov2, shape, b, nt in runs:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            cfg, enc, dec, losses = build(dinov2, shape, n_cand)
            if os.environ.get("TRACE") == "1":   # memory on entry/exit of the encoder's main blocks
                def hook(name, when):
                    def f(*_):
                        print(f"  {when:5s} {name:28s} {torch.cuda.memory_allocated() / 2**30:6.1f} GiB "
                              f"(peak {torch.cuda.max_memory_allocated() / 2**30:.1f})", flush=True)
                    return f
                for name in ("backbone", "depth_predictor", "surfel_adaption", "feature_aggregation",
                             "gaussian_head", "gaussian_adapter"):
                    m = getattr(enc, name, None)
                    if isinstance(m, torch.nn.Module):
                        m.register_forward_pre_hook(hook(name, "enter"))
                        m.register_forward_hook(hook(name, "exit"))
            if os.environ.get("EFF") == "1":
                print(f"  windowed attention max |diff| vs original: {check_equivalence(enc):.2e}", flush=True)
            if os.environ.get("CKPT") == "1":
                checkpoint_feature_aggregation(enc)
            opt = torch.optim.Adam([p for p in enc.parameters() if p.requires_grad], lr=1e-4)
            batch = example(cfg, b, nt)
            ts = []
            for i in range(4):
                torch.cuda.synchronize()
                t0 = time.time()
                step(enc, dec, losses, opt, batch, bf16=os.environ.get("BF16") == "1")
                torch.cuda.synchronize()
                ts.append(time.time() - t0)
            peak = torch.cuda.max_memory_allocated() / 2**30
            resv = torch.cuda.max_memory_reserved() / 2**30
            msg = f"peak {peak:5.1f} GiB (reserved {resv:5.1f})  {min(ts[1:]):.2f} s/step"
        except torch.cuda.OutOfMemoryError as e:
            import traceback
            tb = [f"{f.filename.split('/')[-1]}:{f.lineno} {f.line}" for f in traceback.extract_tb(e.__traceback__)][-5:]
            msg = f"OOM at {torch.cuda.memory_allocated() / 2**30:.1f} GiB allocated, in " + " <- ".join(reversed(tb))
        except Exception as e:  # noqa: BLE001
            import traceback
            tb = [f"{f.filename.split('/')[-1]}:{f.lineno} {f.line}" for f in traceback.extract_tb(e.__traceback__)][-4:]
            msg = f"error {type(e).__name__}: {str(e)[:120]} in " + " <- ".join(reversed(tb))
        name = ("dinov2 (released)" if dinov2 else "cost volume") + \
            "".join(f" +{k.lower()}" for k in ("EFF", "CKPT", "BF16") if os.environ.get(k) == "1")
        print(f"{name:28s} {shape[0]}x{shape[1]}  batch {b}  3 ctx + {nt} target"
              f"{f'  D={n_cand}' if n_cand else ''}: {msg}", flush=True)
        for v in ("enc", "dec", "opt", "batch", "losses"):
            locals().pop(v, None)
        enc = dec = opt = batch = losses = None


if __name__ == "__main__":
    main()
