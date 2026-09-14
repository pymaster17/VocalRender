"""
Dynamic batch sampler for VoxCPM training.

Provides :class:`DynamicBatchSampler`, a :class:`torch.utils.data.Sampler`
that groups samples so the total token count per batch stays within a
configurable budget.  For distributed training it plans *global* steps
first (assigning samples to ranks greedily to balance cost) and then
partitions per rank.

Extracted from ``svs_data.py`` to keep batch-planning infrastructure
separate from dataset / collation logic.
"""

import random
from typing import List

import torch
import torch.distributed as dist


class DynamicBatchSampler(torch.utils.data.Sampler):
    """
    Dynamic batch sampler that groups samples based on their total_length.

    Instead of using a fixed batch_size, this sampler creates batches where
    the total number of tokens (sum of all sample lengths) doesn't exceed
    max_batch_tokens. This helps:
    1. Avoid OOM errors on long sequences
    2. Better GPU utilization by packing similar-length samples together

    For distributed training, batches are partitioned across GPUs using
    rank and world_size parameters.

    Implementation note:
        In distributed training this sampler acts as a global-step planner.
        It first constructs a *distributed step* as ``world_size`` local
        micro-batches together, then assigns samples greedily so the costs of
        those local micro-batches stay close. Only after the global steps are
        planned are they shuffled as whole units. This avoids the common
        failure mode where each rank independently gets a valid local batch
        but the same training step is badly imbalanced across ranks.

    Args:
        lengths: Pre-computed list of sequence lengths for each sample
        max_batch_tokens: Maximum total tokens per batch
        max_batch_size: Maximum number of samples per batch (optional cap)
        shuffle: Whether to shuffle samples each epoch
        drop_last: Whether to drop the last incomplete batch
        rank: Process rank for distributed training (0 for single GPU)
        world_size: Total number of processes (1 for single GPU)
        seed: Random seed for reproducible shuffling across processes
    """

    def __init__(
        self,
        lengths: List[int],
        max_batch_tokens: int,
        max_batch_size: int = 64,
        shuffle: bool = True,
        drop_last: bool = False,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
    ):
        self.lengths = lengths
        self.max_batch_tokens = max_batch_tokens
        self.max_batch_size = max_batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0
        self.batches: List[List[int]] | None = None
        self._start_batch = 0

        self._create_batches()

    def set_start_batch(self, n: int) -> None:
        """Skip the first ``n`` batches on the next ``__iter__`` (consumed once).

        Used by the resume path to fast-forward the dataloader to a saved
        position without materializing (decoding) the skipped batches. The
        offset is reset to 0 once an iterator is created, so subsequent epochs
        start from the beginning.
        """
        if self.batches is not None and n > len(self.batches):
            raise RuntimeError(
                f"start_batch={n} exceeds available batches ({len(self.batches)}) "
                f"for epoch={self.epoch}."
            )
        self._start_batch = max(0, int(n))

    def set_epoch(self, epoch: int):
        """Set the epoch for reproducible shuffling.

        Skips replanning when the requested epoch matches the currently
        planned one — resume paths call ``set_epoch(data_epoch)`` right
        after construction, which would otherwise redo an O(N·world_size²)
        pure-Python plan for nothing.
        """
        if epoch == self.epoch and self.batches is not None:
            return
        self.epoch = epoch
        self._create_batches()

    def _batch_max_len(self, batch: List[int]) -> int:
        if not batch:
            return 0
        return max(self.lengths[idx] for idx in batch)

    def _batch_cost(self, batch: List[int]) -> int:
        if not batch:
            return 0
        # Attention/activation cost tracks sequence length worse than a simple
        # token budget. Use B * L^2 as a better proxy so that the same
        # distributed step sees similarly expensive batches on every rank.
        max_len = self._batch_max_len(batch)
        return len(batch) * max_len * max_len

    @staticmethod
    def _batch_tokens_from_stats(count: int, max_len: int) -> int:
        if count <= 0 or max_len <= 0:
            return 0
        return count * max_len

    @staticmethod
    def _batch_cost_from_stats(count: int, max_len: int) -> int:
        if count <= 0 or max_len <= 0:
            return 0
        return count * max_len * max_len

    def _order_indices(self, rng: random.Random) -> List[int]:
        indices = list(range(len(self.lengths)))
        if not self.shuffle:
            return indices

        # Sort by length first, then shuffle within relatively small
        # sortish buckets. Smaller buckets keep lengths tighter than the
        # previous coarse bucketing, which reduces bad local orderings that
        # later force imbalanced distributed steps.
        sorted_indices = sorted(indices, key=lambda i: self.lengths[i])
        chunk_size = max(64, self.world_size * max(1, self.max_batch_size) * 8)
        chunk_size = min(chunk_size, 512)
        chunks = [
            sorted_indices[i:i + chunk_size]
            for i in range(0, len(sorted_indices), chunk_size)
        ]
        for chunk in chunks:
            rng.shuffle(chunk)
        rng.shuffle(chunks)
        return [idx for chunk in chunks for idx in chunk]

    def _candidate_rank_key(
        self,
        *,
        rank: int,
        sample_len: int,
        batch_sizes: List[int],
        batch_max_lens: List[int],
        step_idx: int,
    ):
        count = batch_sizes[rank]
        max_len = batch_max_lens[rank]
        new_count = count + 1
        new_max_len = max(max_len, sample_len)
        projected_tokens = new_count * new_max_len

        # Keep singleton overflow behaviour compatible with the previous
        # implementation: if one sample alone exceeds the budget, still allow
        # it as a batch of size 1 so the sample is not dropped.
        if count > 0 and (
            new_count > self.max_batch_size or projected_tokens > self.max_batch_tokens
        ):
            return None

        projected_cost = self._batch_cost_from_stats(new_count, new_max_len)

        simulated_costs = [
            self._batch_cost_from_stats(existing_count, existing_max_len)
            for existing_count, existing_max_len in zip(batch_sizes, batch_max_lens)
        ]
        simulated_costs[rank] = projected_cost
        cost_spread = max(simulated_costs) - min(simulated_costs)
        max_cost = max(simulated_costs)
        total_cost = sum(simulated_costs)

        simulated_tokens = [
            self._batch_tokens_from_stats(existing_count, existing_max_len)
            for existing_count, existing_max_len in zip(batch_sizes, batch_max_lens)
        ]
        simulated_tokens[rank] = projected_tokens
        token_spread = max(simulated_tokens) - min(simulated_tokens)

        # Rotate tie-breaking so the same rank is not always preferred when
        # costs are equal.
        rotated_rank = (rank - step_idx) % max(1, self.world_size)
        return (
            cost_spread,
            max_cost,
            total_cost,
            token_spread,
            projected_cost,
            projected_tokens,
            new_count,
            rotated_rank,
            rank,
        )

    def _plan_global_steps(
        self,
        ordered_indices: List[int],
        rng: random.Random,
    ) -> List[List[List[int]]]:
        if not ordered_indices:
            return []

        global_steps: List[List[List[int]]] = []
        next_index = 0
        step_idx = 0

        while next_index < len(ordered_indices):
            step_batches = [[] for _ in range(self.world_size)]
            batch_sizes = [0] * self.world_size
            batch_max_lens = [0] * self.world_size
            assigned_any = False

            while next_index < len(ordered_indices):
                sample_idx = ordered_indices[next_index]
                sample_len = self.lengths[sample_idx]

                candidates = []
                for rank in range(self.world_size):
                    candidate_key = self._candidate_rank_key(
                        rank=rank,
                        sample_len=sample_len,
                        batch_sizes=batch_sizes,
                        batch_max_lens=batch_max_lens,
                        step_idx=step_idx,
                    )
                    if candidate_key is not None:
                        candidates.append(candidate_key)

                if not candidates:
                    break

                chosen_rank = min(candidates)[-1]
                step_batches[chosen_rank].append(sample_idx)
                batch_sizes[chosen_rank] += 1
                batch_max_lens[chosen_rank] = max(batch_max_lens[chosen_rank], sample_len)
                assigned_any = True
                next_index += 1

            if not assigned_any:
                # Defensive fallback: this should only happen if the dataset is
                # empty, but avoid an infinite loop if constraints are invalid.
                break

            non_empty_ranks = sum(1 for batch in step_batches if batch)
            if self.drop_last and non_empty_ranks < self.world_size:
                break

            global_steps.append(step_batches)
            step_idx += 1

        if self.shuffle:
            rng.shuffle(global_steps)

        return global_steps

    def _create_batches(self):
        """Create per-rank batches via global distributed-step planning.

        The planner is fully deterministic given ``(seed, epoch)`` and its
        output is identical on every rank — so in distributed runs we let
        global rank 0 do the O(N·world_size²) Python plan once and
        broadcast the result. Every rank then extracts its own slice via
        ``self.rank`` (which is the data-parallel rank, not the global
        one), so correctness under FULL_SHARD and HYBRID_SHARD is
        preserved: replicas that share a dp_rank simply pick the same
        slice.
        """
        use_broadcast = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1

        if use_broadcast:
            if dist.get_rank() == 0:
                rng = random.Random(self.seed + self.epoch)
                ordered_indices = self._order_indices(rng)
                global_steps = self._plan_global_steps(ordered_indices, rng)
                payload = [global_steps]
            else:
                payload = [None]
            dist.broadcast_object_list(payload, src=0)
            global_steps = payload[0]
        else:
            rng = random.Random(self.seed + self.epoch)
            ordered_indices = self._order_indices(rng)
            global_steps = self._plan_global_steps(ordered_indices, rng)

        self.batches = [
            step[self.rank]
            for step in global_steps
            if self.rank < len(step) and step[self.rank]
        ]

    def __iter__(self):
        # Note: batches are already shuffled in _create_batches() with deterministic seed.
        # _start_batch (set by set_start_batch for resume) is consumed once, then reset.
        start = self._start_batch
        self._start_batch = 0
        for batch in self.batches[start:]:
            yield batch

    def __len__(self):
        return len(self.batches)
