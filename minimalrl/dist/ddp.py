"""PyTorch distributed data-parallel launcher utilities."""

from __future__ import annotations

import os
from typing import Callable

import torch.multiprocessing as mp


def _worker(rank: int, world_size: int, fn: Callable[[int, int], None]) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    import torch.distributed as dist

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    try:
        fn(rank, world_size)
    finally:
        dist.destroy_process_group()


def launch_ddp(world_size: int, fn: Callable[[int, int], None]) -> None:
    """Launch a multi-process training job."""

    mp.spawn(_worker, args=(world_size, fn), nprocs=world_size, join=True)
