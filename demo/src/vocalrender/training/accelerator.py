from __future__ import annotations

import contextlib
import os
import random
import typing

import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data
from torch.nn.parallel import DistributedDataParallel


class Accelerator:
    """
    Simplified accelerator that mirrors the behaviour of the minicpm-audio
    training utilities. It initializes a distributed process group when
    ``torchrun`` is used and exposes helpers for AMP, gradient scaling and
    preparing models/dataloaders for DDP.

    Supports optional ZeRO-3 memory optimization via PyTorch FSDP2
    (``fully_shard`` / ``FSDPModule`` — per-parameter DTensor sharding).

    Hybrid sharding
    ---------------
    When ``fsdp_sharding_group_size > 1`` (requires ``zero_stage >= 2``),
    HYBRID_SHARD is used: ZeRO-3 *within* each shard group of consecutive
    local ranks, and DDP all-reduce *between* groups.

    Rank-to-group assignment is purely by rank ordering:
      group 0 → local ranks [0 … group_size-1]
      group 1 → local ranks [group_size … 2*group_size-1]
      …

    The caller is responsible for ensuring that each group contains
    NVLink-connected GPUs.  The standard way to do this is to set
    ``CUDA_VISIBLE_DEVICES`` in the job submission script so that
    NVLink peers are listed consecutively, e.g.:

        # If NVLink pairs are (GPU4,GPU5) and (GPU6,GPU7):
        export CUDA_VISIBLE_DEVICES=4,5,6,7   # → groups {4,5} and {6,7}  ✓
        # NOT: 4,6,5,7                         # → groups {4,6} and {5,7}  ✗
    """

    def __init__(
        self,
        amp: bool = False,
        seed: int = 42,
        amp_dtype: torch.dtype | None = None,
        zero_stage: int = 0,
        fsdp_sharding_group_size: int = 1,
    ):
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)

        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        if zero_stage not in (0, 3):
            raise ValueError(
                f"Unsupported zero_stage={zero_stage}. Expected 0 or 3 "
                "(FSDP2 only supports FULL_SHARD)."
            )
        if fsdp_sharding_group_size < 1:
            raise ValueError(
                f"fsdp_sharding_group_size must be >= 1, got {fsdp_sharding_group_size}"
            )
        if fsdp_sharding_group_size > 1 and zero_stage < 2:
            raise ValueError(
                "fsdp_sharding_group_size > 1 requires zero_stage >= 2"
            )
        if fsdp_sharding_group_size > 1 and self.world_size % fsdp_sharding_group_size != 0:
            raise ValueError(
                f"world_size ({self.world_size}) must be divisible by "
                f"fsdp_sharding_group_size ({fsdp_sharding_group_size})"
            )
        self.fsdp_sharding_group_size = fsdp_sharding_group_size

        if self.world_size > 1 and not dist.is_initialized():
            import datetime
            # Bind the PG explicitly to this rank's GPU. Without device_id NCCL
            # "guesses device based on global rank" (PG-default-device warning)
            # and can deadlock during bring-up — observed as a 2-GPU hang where
            # ranks spun at 100% util / pre-model VRAM. Passing device_id makes
            # init eager + device-correct.
            init_kwargs = dict(
                backend="nccl", init_method="env://",
                timeout=datetime.timedelta(hours=1),
            )
            if torch.cuda.is_available():
                init_kwargs["device_id"] = torch.device("cuda", self.local_rank)
            dist.init_process_group(**init_kwargs)

        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.amp = amp
        self.amp_dtype = amp_dtype
        self.zero_stage = zero_stage

        # Set random seed to ensure model initialization consistency
        self._set_seed(seed)

        class DummyScaler:
            def step(self, optimizer):
                optimizer.step()

            def scale(self, loss):
                return loss

            def unscale_(self, optimizer):
                return optimizer

            def update(self):
                pass

        use_grad_scaler = (
            amp
            and torch.cuda.is_available()
            and self.amp_dtype == torch.float16
        )
        self.scaler = torch.amp.GradScaler("cuda") if use_grad_scaler else DummyScaler()
        self.device_ctx = (
            torch.cuda.device(self.local_rank) if torch.cuda.is_available() else None
        )
        self._ddp_model = None  # For no_sync support
        self._fsdp_model = None  # FSDP2 root module
        self._device_mesh = None  # For HYBRID_SHARD device mesh

    def _set_seed(self, seed: int):
        """Set random seed to ensure model initialization consistency across multiple GPUs"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def __enter__(self):
        if self.device_ctx is not None:
            self.device_ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.device_ctx is not None:
            self.device_ctx.__exit__(exc_type, exc_value, traceback)

    @property
    def is_fsdp(self) -> bool:
        """Whether the model is wrapped with FSDP."""
        return self._fsdp_model is not None

    @property
    def dp_world_size(self) -> int:
        """
        Effective data-parallel world size.

        For standard / ZeRO-3: equals world_size (every rank holds a
        different micro-batch).
        For HYBRID_SHARD: equals world_size // fsdp_sharding_group_size — all
        ranks inside the same shard group compute the same micro-batch, so
        only one representative per group contributes to data partitioning.
        """
        return self.world_size // self.fsdp_sharding_group_size

    @property
    def dp_rank(self) -> int:
        """
        Effective data-parallel rank (index of this replica group).

        For standard modes: same as rank.
        For HYBRID_SHARD: rank // fsdp_sharding_group_size.
        """
        return self.rank // self.fsdp_sharding_group_size

    @property
    def is_hybrid_shard(self) -> bool:
        """True when using HYBRID_SHARD (ZeRO-3 intra-group + DDP inter-group)."""
        return self.fsdp_sharding_group_size > 1 and self.zero_stage >= 2

    def barrier(self):
        """Synchronize all processes"""
        if dist.is_initialized():
            dist.barrier()

    def all_reduce(self, tensor: torch.Tensor, op=dist.ReduceOp.AVG):
        """All-reduce tensor across processes"""
        if dist.is_initialized():
            dist.all_reduce(tensor, op=op)
        return tensor

    # ------------------------------------------------------------------ #
    # Model helpers
    # ------------------------------------------------------------------ #
    def prepare_model(self, model: torch.nn.Module, **kwargs):
        if hasattr(model, 'device'):  # make sure the matrix will be moved to the correct device
            model.device = self.device
        model = model.to(self.device)
        if self.world_size > 1:
            if self.zero_stage >= 2:
                model = self._wrap_fsdp2(model)
                self._fsdp_model = model
            else:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                model = DistributedDataParallel(model, device_ids=[self.local_rank], **kwargs)
                self._ddp_model = model
        return model

    # ---- FSDP2 wrapping ------------------------------------------------ #
    def _wrap_fsdp2(self, model: torch.nn.Module):
        """
        Apply FSDP2 (``fully_shard``) layer-by-layer to MiniCPM decoder
        layers, then to the root module. Per-parameter DTensor sharding.
        """
        from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
        from torch.distributed.device_mesh import init_device_mesh
        from vocalrender.modules.minicpm4.model import MiniCPMDecoderLayer

        if self.is_hybrid_shard:
            n_replicas = self.world_size // self.fsdp_sharding_group_size
            mesh = init_device_mesh(
                "cuda",
                (n_replicas, self.fsdp_sharding_group_size),
                mesh_dim_names=("replicate", "shard"),
            )
        else:
            mesh = init_device_mesh("cuda", (self.world_size,))
        self._device_mesh = mesh

        # Mixed precision is handled entirely via ``torch.amp.autocast`` at the
        # call sites. Leaving FSDP2 with default policy (no MixedPrecisionPolicy)
        # keeps param / grad / reduce dtypes as-stored and avoids an extra
        # cast graph that interacts poorly with activation checkpointing
        # recompute + autocast bf16.
        fsdp_kwargs = {"mesh": mesh}
        _ = MixedPrecisionPolicy  # silence unused import (kept for future use)

        for module in model.modules():
            if isinstance(module, MiniCPMDecoderLayer):
                fully_shard(module, **fsdp_kwargs)
        fully_shard(model, **fsdp_kwargs)

        return model

    @contextlib.contextmanager
    def no_sync(self):
        """
        Context manager to skip gradient synchronization during gradient accumulation.
        Only used outside the last micro-batch.
        For FSDP we intentionally keep sync enabled: ``no_sync()`` forces
        gradients to accumulate unsharded, which can recreate a full-gradient
        memory spike and defeat ZeRO memory savings.
        """
        if self._fsdp_model is not None:
            yield
            return

        if self._ddp_model is not None:
            with self._ddp_model.no_sync():
                yield
            return

        yield

    @contextlib.contextmanager
    def summon_full_params(self, model: torch.nn.Module, writeback: bool = False):
        """
        Context manager that ensures FSDP parameters are gathered (un-sharded)
        for the duration of the block.

        Required whenever calling methods on the *unwrapped* inner module directly
        (e.g. ``generate_batch``), because those calls bypass FSDP's ``forward``
        hook that normally gathers parameters before the forward pass.

        For non-FSDP models this is a no-op. ``writeback`` is accepted for API
        compatibility but has no effect under FSDP2 — mutations to the full
        params during the block are not propagated back to the shards.
        """
        if self._fsdp_model is None:
            yield
            return

        from torch.distributed.fsdp import FSDPModule

        fsdp_modules = [m for m in model.modules() if isinstance(m, FSDPModule)]

        # ``unshard()`` alone is not enough for inference paths that mix a
        # prefill ``forward`` (runs through FSDP's forward hooks) with a
        # subsequent autoregressive ``forward_step`` (bypasses hooks): the
        # prefill's post-forward hook re-shards every FSDP unit's params
        # back to DTensors, so when ``forward_step`` later reads
        # ``self.weight`` it sees a sharded DTensor and mixes it with plain
        # KV-cache tensors → ``aten.mul.Tensor got mixed torch.Tensor and
        # DTensor``.
        #
        # Fix: flip ``reshard_after_forward`` to False on every FSDP unit
        # for the duration of the summon block. With resharding disabled,
        # the prefill forward keeps params unsharded all the way through
        # the generation loop. On exit, restore the original post-forward
        # mesh info and call ``reshard()`` explicitly so training resumes
        # with the normal sharded state.
        saved_post_forward_info = []
        for m in fsdp_modules:
            state = m._get_fsdp_state()
            pg = state._fsdp_param_group
            saved_post_forward_info.append(
                (m, state._auto_reshard_after_forward,
                 pg.post_forward_mesh_info if pg is not None else None)
            )
            m.set_reshard_after_forward(False, recurse=False)

        for m in fsdp_modules:
            m.unshard()
        try:
            yield
        finally:
            for m, auto_flag, post_info in saved_post_forward_info:
                state = m._get_fsdp_state()
                state._auto_reshard_after_forward = auto_flag
                pg = state._fsdp_param_group
                if pg is not None:
                    pg.post_forward_mesh_info = post_info
            for m in fsdp_modules:
                m.reshard()

    @property
    def device(self):
        if torch.cuda.is_available():
            return torch.device("cuda", self.local_rank)
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    # ------------------------------------------------------------------ #
    # AMP helpers
    # ------------------------------------------------------------------ #
    def autocast(self, *args, **kwargs):
        if "dtype" not in kwargs and self.amp_dtype is not None:
            kwargs["dtype"] = self.amp_dtype
        return torch.amp.autocast("cuda", enabled=self.amp, *args, **kwargs)

    def backward(self, loss: torch.Tensor):
        self.scaler.scale(loss).backward()

    def step(self, optimizer: torch.optim.Optimizer):
        self.scaler.step(optimizer)

    def update(self):
        self.scaler.update()

    # ------------------------------------------------------------------ #
    # Gradient helpers
    # ------------------------------------------------------------------ #
    def clip_grad_norm(self, model: torch.nn.Module, max_norm: float) -> torch.Tensor:
        """Clip gradient norm. FSDP2 grads are DTensors; PT's clip_grad_norm_
        handles them and reduces across all mesh dims internally."""
        if self._fsdp_model is not None:
            return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        unwrapped = self.unwrap(model)
        return torch.nn.utils.clip_grad_norm_(unwrapped.parameters(), max_norm=max_norm)

    # ------------------------------------------------------------------ #
    # State dict helpers (FSDP-aware)
    # ------------------------------------------------------------------ #
    def model_state_dict(self, model: torch.nn.Module) -> dict | None:
        """
        Get full model state dict.

        For FSDP: collective operation (all ranks MUST call). Returns the full
        state dict on rank 0, empty dict on other ranks.
        For DDP: returns full state dict on rank 0, None on other ranks.
        """
        if self._fsdp_model is not None:
            from torch.distributed.checkpoint.state_dict import (
                get_model_state_dict,
                StateDictOptions,
            )
            opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
            return get_model_state_dict(model, options=opts)
        if self.rank != 0:
            return None
        return self.unwrap(model).state_dict()

    def load_model_state_dict(
        self, model: torch.nn.Module, state_dict: dict, **kwargs
    ):
        """
        Load state dict into model. Handles FSDP transparently.

        For FSDP: collective operation (all ranks MUST call).
        For DDP: standard load on the unwrapped model.
        """
        if self._fsdp_model is not None:
            from torch.distributed.checkpoint.state_dict import (
                set_model_state_dict,
                StateDictOptions,
            )
            # broadcast_from_rank0 lets us pass a full state_dict on rank 0
            # only (others can pass {}); strict=False mirrors prior behavior.
            opts = StateDictOptions(
                full_state_dict=True,
                broadcast_from_rank0=True,
                strict=kwargs.get("strict", True),
            )
            set_model_state_dict(model, state_dict, options=opts)
            return
        self.unwrap(model).load_state_dict(state_dict, **kwargs)

    def optimizer_state_dict(
        self, model: torch.nn.Module, optimizer: torch.optim.Optimizer
    ) -> dict | None:
        """
        Get full optimizer state dict.

        For FSDP: collective operation (all ranks MUST call). Returns full
        state on rank 0, empty on others.
        For DDP: returns state dict on rank 0 only.
        """
        if self._fsdp_model is not None:
            from torch.distributed.checkpoint.state_dict import (
                get_optimizer_state_dict,
                StateDictOptions,
            )
            opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
            return get_optimizer_state_dict(model, optimizer, options=opts)
        if self.rank != 0:
            return None
        return optimizer.state_dict()

    def load_optimizer_state_dict(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        state_dict: dict,
    ):
        """
        Load optimizer state dict. Handles FSDP transparently.

        For FSDP: collective operation (all ranks MUST call).
        For DDP: standard load.
        """
        if self._fsdp_model is not None:
            from torch.distributed.checkpoint.state_dict import (
                set_optimizer_state_dict,
                StateDictOptions,
            )
            opts = StateDictOptions(
                full_state_dict=True,
                broadcast_from_rank0=True,
            )
            set_optimizer_state_dict(
                model, optimizer, state_dict, options=opts
            )
            return
        optimizer.load_state_dict(state_dict)

    # ------------------------------------------------------------------ #
    # Data helpers
    # ------------------------------------------------------------------ #
    def prepare_dataloader(
        self,
        dataset: typing.Iterable,
        *,
        batch_size: int,
        num_workers: int = 0,
        shuffle: bool = True,
        collate_fn=None,
        drop_last: bool = False,
    ) -> torch.utils.data.DataLoader:
        if self.world_size > 1:
            sampler_num_replicas = self.dp_world_size if self.is_hybrid_shard else self.world_size
            sampler_rank = self.dp_rank if self.is_hybrid_shard else self.rank
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=sampler_num_replicas,
                rank=sampler_rank,
                shuffle=shuffle,
            )
            shuffle = False
        else:
            sampler = None

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            drop_last=drop_last,
            pin_memory=True,
        )

    @staticmethod
    def unwrap(model: torch.nn.Module) -> torch.nn.Module:
        """Unwrap DDP wrapper to access the original module.

        FSDP2 (``fully_shard``) is applied in-place — there is no wrapper, so
        the model object IS the unwrapped module. Only DDP needs unwrapping.
        """
        if hasattr(model, "module"):
            return model.module
        return model
