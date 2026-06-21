"""m-per-cosmology batch sampler for the VICReg invariance term.

The VICReg redesign (Williamson et al. DES Y3 arXiv:2606.11309 §3.4, eq.13) requires the
invariance term to pull together summaries of DIFFERENT REALISATIONS of the SAME cosmology θ —
not two augmentations of one map. To get same-cosmology positives in every batch with a SINGLE
encoder forward (no doubled map reads), this sampler packs each batch with ``k`` distinct
cosmologies, each contributing ``m`` realisations (``k * m == batch_size``). The VICReg
LightningModule then groups the batch rows by the per-sample cosmology id and computes the
invariance term over same-cosmology members (SupCon-style m-per-class, research Option 2).

Design choices (see the task plan):
- **Fixed-length epochs:** ``len = n_samples // batch_size`` so every batch is full-size (k
  distinct cosmologies × m) and the variance/covariance terms always see a fixed B. Coverage is
  approximately uniform across an epoch; a sample may repeat / be skipped within an epoch (fine
  for SBI pre-training with ~10^5 files and arbitrary epoch boundaries).
- **Distinct cosmologies per batch:** cosmologies are drawn from a shuffled queue, ``k`` at a
  time; the queue is refilled (reshuffled) only when fewer than ``k`` remain, so a batch never
  contains the same cosmology twice. Requires ``k <= n_cosmologies`` (asserted).
- **DDP-aware:** when ``num_replicas > 1`` the cosmologies are sharded disjointly across ranks
  (``cosmos[rank::num_replicas]``), mirroring split-by-cosmology's no-leakage ethos at the batch
  level. Single-GPU (the only path on this cluster's hybrids) resolves to world=1/rank=0.
- **Per-epoch reshuffle:** an internal epoch counter (advanced each ``__iter__``) reseeds the RNG
  so ``persistent_workers`` still reshuffle. The batch_sampler is iterated in the MAIN process, so
  this works without an external ``set_epoch`` call.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence

import torch
import torch.distributed as dist
from torch.utils.data import Sampler

from .data_selection import extract_cosmo_index


class MPerCosmoBatchSampler(Sampler):
    """Yield lists of dataset indices: ``k`` distinct cosmologies × ``m`` realisations each."""

    def __init__(
        self,
        paths: Sequence[str],
        m_per_cosmo: int,
        batch_size: int,
        *,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 42,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
    ):
        if m_per_cosmo < 1:
            raise ValueError(f"m_per_cosmo must be >= 1, got {m_per_cosmo}")
        if batch_size % m_per_cosmo != 0:
            raise ValueError(
                f"batch_size ({batch_size}) must be divisible by m_per_cosmo ({m_per_cosmo})"
            )
        self.m = int(m_per_cosmo)
        self.batch_size = int(batch_size)
        self.k = self.batch_size // self.m
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)  # kept for API symmetry; epochs are fixed-length
        self.seed = int(seed)
        self.epoch = 0

        # Resolve DDP topology: explicit args win, else torch.distributed if initialised, else 1/0.
        if num_replicas is None or rank is None:
            if dist.is_available() and dist.is_initialized():
                num_replicas = dist.get_world_size()
                rank = dist.get_rank()
            else:
                num_replicas = 1
                rank = 0
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)

        # Group dataset indices (positions in `paths`) by integer cosmology id.
        by_cosmo: Dict[int, List[int]] = {}
        for i, p in enumerate(paths):
            cid = extract_cosmo_index(p)
            by_cosmo.setdefault(cid, []).append(i)
        all_cosmos = sorted(by_cosmo.keys())
        # DDP sharding: disjoint cosmologies per rank (no cross-rank cosmology overlap).
        my_cosmos = all_cosmos[self.rank :: self.num_replicas]
        self.by_cosmo: Dict[int, List[int]] = {c: by_cosmo[c] for c in my_cosmos}
        self.cosmos: List[int] = list(self.by_cosmo.keys())
        if len(self.cosmos) < self.k:
            raise ValueError(
                f"MPerCosmoBatchSampler: k = batch_size//m = {self.k} exceeds the number of "
                f"available cosmologies ({len(self.cosmos)}) on rank {self.rank} of "
                f"{self.num_replicas}. Lower batch_size or m_per_cosmo, or add more cosmologies."
            )
        self.n_samples = sum(len(v) for v in self.by_cosmo.values())
        self._num_batches = self.n_samples // self.batch_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self._num_batches

    def __iter__(self):
        # Advance the epoch each call so persistent_workers still reshuffle across epochs.
        self.epoch += 1
        g = random.Random(self.seed + self.epoch) if self.shuffle else random.Random(self.seed)

        # Per-cosmology cyclic index pools (reshuffled on wraparound).
        pools: Dict[int, List[int]] = {}
        pos: Dict[int, int] = {}
        for c, idxs in self.by_cosmo.items():
            lst = list(idxs)
            if self.shuffle:
                g.shuffle(lst)
            pools[c] = lst
            pos[c] = 0

        def draw(c: int) -> List[int]:
            lst = pools[c]
            # A cosmology with fewer than m realisations: sample with replacement to fill m
            # (guarantees >= 1 positive pair; degenerate cosmologies still contribute).
            if len(lst) < self.m:
                return [lst[g.randrange(len(lst))] for _ in range(self.m)]
            p = pos[c]
            if p + self.m > len(lst):
                if self.shuffle:
                    g.shuffle(lst)
                p = 0
            out = lst[p : p + self.m]
            pos[c] = p + self.m
            return out

        def refill_queue() -> List[int]:
            q = list(self.cosmos)
            if self.shuffle:
                g.shuffle(q)
            return q

        cosmo_queue: List[int] = refill_queue()
        for _ in range(self._num_batches):
            if len(cosmo_queue) < self.k:
                cosmo_queue = refill_queue()
            chosen = cosmo_queue[: self.k]
            cosmo_queue = cosmo_queue[self.k :]
            batch: List[int] = []
            for c in chosen:
                batch.extend(draw(c))
            yield batch
