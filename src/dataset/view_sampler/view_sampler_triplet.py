from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float, Int64
from torch import Tensor

from .view_sampler import ViewSampler


@dataclass
class ViewSamplerTripletCfg:
    """Three context views [c - g, c, c + g] from an ordered sequence, and
    target views drawn from the same span (preferring non-context frames)."""
    name: Literal["triplet"]
    num_context_views: int
    num_target_views: int
    min_gap: int
    max_gap: int


class ViewSamplerTriplet(ViewSampler[ViewSamplerTripletCfg]):
    def sample(
        self,
        scene: str,
        extrinsics: Float[Tensor, "view 4 4"],
        intrinsics: Float[Tensor, "view 3 3"],
        device: torch.device = torch.device("cpu"),
    ) -> tuple[
        Int64[Tensor, " context_view"],
        Int64[Tensor, " target_view"],
    ]:
        n = extrinsics.shape[0]
        max_gap = min(self.cfg.max_gap, (n - 1) // 2)
        if max_gap < self.cfg.min_gap:
            raise ValueError("Example does not have enough frames!")
        if self.stage == "test":
            gap, centre = max_gap, n // 2
        else:
            gap = int(torch.randint(self.cfg.min_gap, max_gap + 1, ()).item())
            centre = int(torch.randint(gap, n - gap, ()).item())
        context = torch.tensor([centre - gap, centre, centre + gap], dtype=torch.int64, device=device)
        span = torch.arange(centre - gap, centre + gap + 1, device=device)
        pool = span[~torch.isin(span, context)] if gap > 1 else span
        target = pool[torch.randint(len(pool), (self.cfg.num_target_views,), device=device)]
        return context, target

    @property
    def num_context_views(self) -> int:
        return 3

    @property
    def num_target_views(self) -> int:
        return self.cfg.num_target_views
