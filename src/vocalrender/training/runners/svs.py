from __future__ import annotations

import contextlib
import os
import time
from pathlib import Path
from typing import Dict

import torch

from vocalrender.training import Accelerator
from vocalrender.training.checkpoint import save_checkpoint
from vocalrender.training.config import SVSTrainConfig, load_svs_train_config
from vocalrender.training.dataset_ops import (
    build_song_index,
    build_val_prompt_pool,
    cap_dataset_hours,
    filter_train_datasets,
    upsample_datasets,
)
from vocalrender.training.diagnostics import (
    format_rank_diag_summary,
    gather_rank_diag_interval,
    log_cuda_memory_breakdown_once,
    log_cuda_peak_memory_once,
    new_rank_diag_interval,
    sync_device_for_timing,
    update_rank_diag_interval,
)
from vocalrender.training.loop_schedule import build_svs_eval_schedule, should_save_checkpoint
from vocalrender.training.model_factory import detect_model_classes
from vocalrender.training.resume import (
    gather_global_sample_progress,
    restore_local_runtime_state,
    restore_train_iterator,
    set_train_loader_epoch,
)
from vocalrender.training.runtime import (
    dtype_name,
    resolve_training_precision,
    LoopProgressState,
    build_optimizer_and_scheduler,
    close_training_runtime,
    create_training_runtime,
    load_resume_context,
    save_training_checkpoint,
)
from vocalrender.training.svs_data import (
    build_preprocessed_svs_dataloader,
    load_preprocessed_svs_datasets,
)
from vocalrender.training.svs_eval_setup import setup_svs_eval
from vocalrender.training.svs_tokenizer import (
    collect_svs_mask_token_ids,
    extend_tokenizer_for_svs,
)
from vocalrender.training.validation import validate_svs


PROJECT_ROOT = Path(__file__).resolve().parents[4]


class SVSRunner:
    def __init__(self, config: SVSTrainConfig):
        self.config = config
        self.accelerator = None
        self.runtime = None
        self.precision_cfg = None
        self.audio_eval_interval = None
        self.loop_state = LoopProgressState()
        self.train_iter = None
        self.resume_context = None

        self.base_model = None
        self.base_tokenizer = None
        self.text_tokenizer = None
        self.mask_ids = None
        self.model = None
        self.unwrapped_model = None
        self.train_loader = None
        self.val_loader = None
        self.train_ds = None
        self.val_ds = None
        self.num_train_samples = 0
        self.prompt_pool_ds = None
        self.prompt_pool_song_index = None
        self.val_song_index = None
        self.val_offset = None
        self.evaluator = None
        self.eval_ref_cache = None
        self.load_audio_vae = None
        self.optimizer = None
        self.scheduler = None
        self.has_logged_cuda_memory = False
        self.timing_interval = None
        self.rank_diag_interval = None

    @property
    def tracker(self):
        return self.runtime.tracker

    @property
    def writer(self):
        return self.runtime.writer

    def build_runtime(self) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        torch.set_float32_matmul_precision("high")

        self._sdpa_flash_disabled = False
        if self.config.dist.zero_stage >= 2 and self.config.model.activation_checkpointing:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
            self._sdpa_flash_disabled = True

        if self.config.model.lora is not None and self.config.model.distribute and not self.config.model.hf_model_id:
            raise ValueError("hf_model_id is required when distribute=True")

        self.precision_cfg = resolve_training_precision(self.config.dist.train_precision)
        self.accelerator = Accelerator(
            amp=self.precision_cfg["amp"],
            amp_dtype=self.precision_cfg["amp_dtype"],
            zero_stage=self.config.dist.zero_stage,
            fsdp_sharding_group_size=self.config.dist.fsdp_sharding_group_size,
        )

        if self.config.train.valid_interval <= 0:
            raise ValueError(f"valid_interval must be > 0, got {self.config.train.valid_interval}")
        audio_eval_interval = self.config.train.audio_eval_interval
        if audio_eval_interval == 0 or audio_eval_interval < -1:
            raise ValueError(
                "audio_eval_interval must be -1 (auto) or a positive integer, "
                f"got {audio_eval_interval}"
            )
        if audio_eval_interval < 0:
            audio_eval_interval = self.config.train.valid_interval * 4
        self.audio_eval_interval = audio_eval_interval

        self.runtime = create_training_runtime(
            accelerator=self.accelerator,
            save_path=self.config.runtime.save_path,
            tensorboard=self.config.runtime.tensorboard,
        )

        if self.accelerator.rank != 0:
            return

        self.tracker.print(
            "Training precision: "
            f"{self.precision_cfg['mode']} "
            f"({self.precision_cfg['description']}; "
            f"param_dtype={dtype_name(self.precision_cfg['param_dtype'])}, "
            f"amp_dtype={dtype_name(self.precision_cfg['amp_dtype'])})"
        )
        self.tracker.print(
            f"Validation schedule: val_loss every {self.config.train.valid_interval} step(s), "
            f"audio eval every {self.audio_eval_interval} step(s)"
        )
        if self.config.runtime.timing_enabled:
            self.tracker.print(
                "Timing instrumentation: enabled "
                f"(rank0, averaged over log_interval={self.config.train.log_interval}, CUDA synchronized for accuracy)"
            )
        if self.config.runtime.per_rank_diagnostics and self.accelerator.world_size > 1:
            self.tracker.print(
                "Per-rank diagnostics: enabled "
                f"(all ranks timed, summary every log_interval={self.config.train.log_interval})"
            )
        if self._sdpa_flash_disabled:
            self.tracker.print("SDPA backend: flash disabled, mem_efficient + math enabled")

        zero_stage = self.config.dist.zero_stage
        if zero_stage >= 2:
            n_groups = self.accelerator.dp_world_size
            g_size = self.accelerator.fsdp_sharding_group_size
            if self.accelerator.is_hybrid_shard:
                inner = "FULL_SHARD" if zero_stage >= 3 else "SHARD_GRAD_OP (ZeRO-2)"
                self.tracker.print(
                    f"Memory optimization: ZeRO-{zero_stage} HYBRID "
                    f"({inner} within {n_groups} groups of {g_size} GPUs each; "
                    f"DDP all-reduce between groups)"
                )
            elif zero_stage >= 3:
                self.tracker.print(
                    f"Memory optimization: ZeRO-{zero_stage} "
                    f"(FSDP FULL_SHARD — parameters, gradients & optimizer states "
                    f"sharded across {self.accelerator.world_size} GPUs)"
                )
            else:
                self.tracker.print(
                    f"Memory optimization: ZeRO-{zero_stage} "
                    f"(FSDP SHARD_GRAD_OP — optimizer states & gradients "
                    f"sharded across {self.accelerator.world_size} GPUs)"
                )
        else:
            self.tracker.print("Memory optimization: none (standard DDP)")

    def build_model_and_tokenizer(self) -> None:
        if self.accelerator.rank == 0:
            self.tracker.print("Loading base model...")
        mc = detect_model_classes(self.config.model.pretrained_path)
        if self.accelerator.rank == 0:
            self.tracker.print(f"Detected architecture: {mc.arch} -> {mc.model_cls.__name__}")
        base_model = mc.model_cls.from_local(
            self.config.model.pretrained_path,
            training=True,
            lora_config=mc.lora_config_cls(**self.config.model.lora) if self.config.model.lora else None,
        )
        tokenizer_wrapper = base_model.text_tokenizer
        self.text_tokenizer = tokenizer_wrapper

        if self.accelerator.rank == 0:
            self.tracker.print("Extending tokenizer with SVS tokens...")
        tok_artifacts = extend_tokenizer_for_svs(base_model, tokenizer_wrapper)
        self.base_tokenizer = tok_artifacts.base_tokenizer
        if self.accelerator.rank == 0:
            self.tracker.print(
                f"Added {tok_artifacts.num_added} SVS tokens to tokenizer "
                f"(new vocab size: {len(self.base_tokenizer)})"
            )

        target_param_dtype = self.precision_cfg["param_dtype"]
        base_model = base_model.to(target_param_dtype if target_param_dtype != torch.float32 else torch.float32)
        base_model.audio_vae = base_model.audio_vae.to(torch.float32)
        if self.accelerator.rank == 0:
            self.tracker.print(
                "Model storage dtype configured: "
                f"{dtype_name(target_param_dtype)} (AudioVAE kept in float32)"
            )

        if self.config.model.activation_checkpointing:
            if not hasattr(base_model, "set_activation_checkpointing"):
                raise AttributeError("Model does not support activation checkpointing")
            base_model.set_activation_checkpointing(True)
            if self.accelerator.rank == 0:
                self.tracker.print(
                    "Activation checkpointing: enabled "
                    "(model-internal single-output closure per MiniCPM block)"
                )

        mask_strategy = self.config.data.svs_mask_strategy
        masking_enabled = (
            (isinstance(mask_strategy, list) and len(mask_strategy) > 0)
            or (isinstance(mask_strategy, str) and mask_strategy != "none")
        )
        self.mask_ids = {
            "pitch_token_ids": set(),
            "note_token_ids": set(),
            "bpm_token_ids": set(),
            "mask_token_id": 0,
        }
        if masking_enabled:
            mask_ids = collect_svs_mask_token_ids(self.base_tokenizer, tok_artifacts)
            self.mask_ids = {
                "pitch_token_ids": mask_ids.pitch_token_ids,
                "note_token_ids": mask_ids.note_token_ids,
                "bpm_token_ids": mask_ids.bpm_token_ids,
                "mask_token_id": mask_ids.mask_token_id,
            }
            if self.accelerator.rank == 0 and masking_enabled:
                self.tracker.print(
                    f"SVS masking: strategies={mask_strategy}, params={self.config.data.svs_mask_params}"
                )
                self.tracker.print(
                    "  pitch_token_ids: "
                    f"{len(self.mask_ids['pitch_token_ids'])}, "
                    f"note_token_ids: {len(self.mask_ids['note_token_ids'])}, "
                    f"bpm_token_ids: {len(self.mask_ids['bpm_token_ids'])}, "
                    f"mask_token_id: {self.mask_ids['mask_token_id']}"
                )

        self.base_model = base_model

    def _resolve_repo_path(self, raw_path: str) -> Path:
        path = Path(raw_path)
        if path.is_absolute():
            return path
        return PROJECT_ROOT / path

    def build_datasets_and_loaders(self) -> None:
        if self.accelerator.rank == 0:
            self.tracker.print(f"Loading PREPROCESSED SVS data from: {self.config.data.preprocessed_data_path}")

        train_ds, val_ds = load_preprocessed_svs_datasets(
            self.config.data.preprocessed_data_path,
            val_datasets=self.config.data.val_datasets,
            val_songs=self.config.data.val_songs,
            val_samples=self.config.data.val_samples,
        )

        log_fn = self.tracker.print if self.accelerator.rank == 0 else None
        if self.config.data.train_datasets:
            train_ds = filter_train_datasets(train_ds, self.config.data.train_datasets, log_fn=log_fn)
        if self.config.data.cloudmusic_max_hours > 0:
            train_ds = cap_dataset_hours(
                train_ds,
                "cloudmusic",
                self.config.data.cloudmusic_max_hours,
                log_fn=log_fn,
            )
        if self.config.data.dataset_upsample:
            train_ds = upsample_datasets(train_ds, self.config.data.dataset_upsample, log_fn=log_fn)

        self.train_ds = train_ds
        self.val_ds = val_ds
        self.num_train_samples = len(train_ds)

        if self.accelerator.rank == 0:
            self.tracker.print(f"Loaded {self.num_train_samples} preprocessed training samples")
            if val_ds is not None:
                self.tracker.print(f"Loaded {len(val_ds)} preprocessed validation samples")

        song_index = None
        self.val_song_index = None
        self.prompt_pool_ds = None
        self.prompt_pool_song_index = None
        self.val_offset = None
        # Same-song prompt-audio segments are needed for
        # ``prompt_audio_prob > 0`` (front prepend).
        if self.config.data.prompt_audio_prob > 0:
            song_index = build_song_index(train_ds, log_fn=log_fn)
            if self.accelerator.rank == 0:
                self.tracker.print(
                    f"  prompt_audio_prob={self.config.data.prompt_audio_prob}, "
                    f"prompt_max_frames={self.config.data.prompt_max_frames}"
                )
            if val_ds is not None:
                (
                    self.prompt_pool_ds,
                    self.prompt_pool_song_index,
                    self.val_song_index,
                    self.val_offset,
                ) = build_val_prompt_pool(train_ds, val_ds, song_index, log_fn=log_fn)

        self.train_loader = build_preprocessed_svs_dataloader(
            train_ds,
            accelerator=self.accelerator,
            batch_size=self.config.train.batch_size,
            num_workers=self.config.train.num_workers,
            drop_last=True,
            max_batch_tokens=self.config.train.max_batch_tokens,
            mask_strategy=self.config.data.svs_mask_strategy,
            mask_params=self.config.data.svs_mask_params or {},
            pitch_token_ids=self.mask_ids["pitch_token_ids"],
            note_token_ids=self.mask_ids["note_token_ids"],
            bpm_token_ids=self.mask_ids["bpm_token_ids"],
            mask_token_id=self.mask_ids["mask_token_id"],
            prompt_audio_prob=self.config.data.prompt_audio_prob,
            prompt_max_frames=self.config.data.prompt_max_frames,
            song_index=song_index,
            text_tokenizer=self.text_tokenizer,
        )
        val_max_batch_tokens = (
            self.config.train.val_max_batch_tokens
            if self.config.train.val_max_batch_tokens >= 0
            else (self.config.train.max_batch_tokens * 4 if self.config.train.max_batch_tokens > 0 else 0)
        )
        val_batch_size = (
            self.config.train.val_batch_size
            if self.config.train.val_batch_size >= 0
            else self.config.train.batch_size * 4
        )
        self.val_loader = (
            build_preprocessed_svs_dataloader(
                val_ds,
                accelerator=self.accelerator,
                batch_size=val_batch_size,
                num_workers=self.config.train.num_workers,
                drop_last=False,
                max_batch_tokens=val_max_batch_tokens,
                pitch_token_ids=self.mask_ids["pitch_token_ids"],
                note_token_ids=self.mask_ids["note_token_ids"],
                bpm_token_ids=self.mask_ids["bpm_token_ids"],
                mask_token_id=self.mask_ids["mask_token_id"],
                prompt_audio_prob=self.config.data.prompt_audio_prob,
                prompt_max_frames=self.config.data.prompt_max_frames,
                song_index=self.val_song_index,
                prompt_audio_seed=42,
                text_tokenizer=self.text_tokenizer,
            )
            if val_ds is not None
            else None
        )

        if self.accelerator.rank == 0 and self.config.train.max_batch_tokens > 0:
            self.tracker.print(f"Using dynamic batching with max_batch_tokens={self.config.train.max_batch_tokens}")
            self.tracker.print(f"  Number of training batches: {len(self.train_loader)}")

        del self.base_model.audio_vae
        trainable_params = sum(p.numel() for p in self.base_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.base_model.parameters())
        self.model = self.accelerator.prepare_model(self.base_model)
        self.unwrapped_model = self.accelerator.unwrap(self.model)
        self.unwrapped_model.train()
        if self.accelerator.rank == 0:
            self.tracker.print(
                f"Trainable parameters: {trainable_params:,} / {total_params:,} "
                f"({100 * trainable_params / total_params:.2f}%)"
            )

    def build_eval_context(self) -> None:
        eval_artifacts = setup_svs_eval(
            self.config.eval.eval_metrics,
            self.config.model.pretrained_path,
            project_root=PROJECT_ROOT,
            sample_rate=self.config.train.sample_rate,
            log_fn=self.tracker.print,
            rank=self.accelerator.rank,
        )
        self.evaluator = eval_artifacts.evaluator
        self.eval_ref_cache = eval_artifacts.eval_ref_cache
        self.load_audio_vae = eval_artifacts.load_audio_vae_fn

    def build_training_state(self) -> None:
        total_training_steps = (
            self.config.train.max_steps if self.config.train.max_steps > 0 else self.config.train.num_iters
        )
        self.optimizer, self.scheduler, scheduler_desc = build_optimizer_and_scheduler(
            self.model,
            learning_rate=self.config.train.learning_rate,
            weight_decay=self.config.train.weight_decay,
            scheduler_type=self.config.train.lr_scheduler,
            warmup_steps=self.config.train.warmup_steps,
            total_training_steps=total_training_steps,
        )
        if self.accelerator.rank == 0:
            self.tracker.print(scheduler_desc)

        self.resume_context = load_resume_context(
            self.model,
            self.optimizer,
            self.scheduler,
            runtime=self.runtime,
        )
        self.loop_state = LoopProgressState()
        if self.resume_context.resume_runtime_state is not None:
            self.loop_state.data_epoch = int(self.resume_context.resume_runtime_state.get("data_epoch", 0))
            self.loop_state.batches_seen_in_epoch = int(
                self.resume_context.resume_runtime_state.get("batches_seen_in_epoch", 0)
            )
            self.loop_state.local_samples_seen = int(
                self.resume_context.resume_runtime_state.get("samples_seen", 0)
            )
            restore_local_runtime_state(self.resume_context.resume_runtime_state, self.accelerator)
            if self.accelerator.rank == 0:
                self.tracker.print(
                    "Restored train dataloader position: "
                    f"epoch={self.loop_state.data_epoch}, "
                    f"batches_seen_in_epoch={self.loop_state.batches_seen_in_epoch}"
                )
        self.train_iter = restore_train_iterator(
            self.train_loader,
            data_epoch=self.loop_state.data_epoch,
            batches_seen_in_epoch=self.loop_state.batches_seen_in_epoch,
        )
        self.has_logged_cuda_memory = False
        self.timing_interval = {
            "data": 0.0,
            "forward": 0.0,
            "backward": 0.0,
            "optim": 0.0,
            "step": 0.0,
            "tokens": 0,
            "micro_steps": 0,
            "steps": 0,
        }
        self.rank_diag_interval = (
            new_rank_diag_interval()
            if self.config.runtime.per_rank_diagnostics and self.accelerator.world_size > 1
            else None
        )

    def get_next_batch(self):
        try:
            batch = next(self.train_iter)
            self.loop_state.batches_seen_in_epoch += 1
            return batch
        except StopIteration:
            self.loop_state.data_epoch += 1
            self.loop_state.batches_seen_in_epoch = 0
            set_train_loader_epoch(self.train_loader, self.loop_state.data_epoch)
            self.train_iter = iter(self.train_loader)
            batch = next(self.train_iter)
            self.loop_state.batches_seen_in_epoch = 1
            return batch

    def train_one_step(self, step: int) -> dict:
        self.resume_context.signal_state["step"] = step
        self.tracker.step = step
        self.optimizer.zero_grad(set_to_none=True)

        timing_active = self.config.runtime.timing_enabled and self.accelerator.rank == 0
        rank_diag_active = self.config.runtime.per_rank_diagnostics and self.accelerator.world_size > 1
        measure_step_timing = timing_active or rank_diag_active

        step_timing = None
        if measure_step_timing:
            sync_device_for_timing(self.accelerator.device)
            step_start_time = time.perf_counter()
            step_timing = {
                "data": 0.0,
                "forward": 0.0,
                "backward": 0.0,
                "optim": 0.0,
                "tokens": 0,
                "micro_steps": 0,
                "max_batch_size": 0,
                "max_seq_len": 0,
                "max_micro_tokens": 0,
            }

        memory_probe = None
        if (
            self.accelerator.rank == 0
            and not self.has_logged_cuda_memory
            and self.accelerator.device.type == "cuda"
            and torch.cuda.is_available()
        ):
            torch.cuda.synchronize(self.accelerator.device)
            memory_probe = {
                "baseline_allocated": torch.cuda.memory_allocated(self.accelerator.device),
                "baseline_reserved": torch.cuda.memory_reserved(self.accelerator.device),
                "forward_peak_allocated": 0,
                "forward_peak_tokens": 0,
                "backward_peak_allocated": 0,
                "backward_peak_tokens": 0,
                "step_end_allocated": 0,
                "step_end_reserved": 0,
            }

        loss_dict = {}
        processed = None
        for micro_step in range(max(int(self.config.train.grad_accum_steps), 1)):
            data_start_time = time.perf_counter() if step_timing is not None else None
            batch = self.get_next_batch()
            processed = {
                "text_tokens": batch["text_tokens"].to(self.accelerator.device),
                "text_mask": batch["text_mask"].to(self.accelerator.device),
                "audio_feats": batch["audio_feats"].to(self.accelerator.device),
                "audio_mask": batch["audio_mask"].to(self.accelerator.device),
                "loss_mask": batch["loss_mask"].to(self.accelerator.device),
                "labels": batch["labels"].to(self.accelerator.device),
            }
            current_batch_tokens = int(processed["text_tokens"].numel())
            current_batch_size = int(processed["text_tokens"].shape[0])
            current_batch_seq_len = int(processed["text_tokens"].shape[1])
            self.loop_state.local_samples_seen += current_batch_size
            if step_timing is not None:
                sync_device_for_timing(self.accelerator.device)
                step_timing["data"] += time.perf_counter() - data_start_time
                step_timing["tokens"] += current_batch_tokens
                step_timing["micro_steps"] += 1
                step_timing["max_batch_size"] = max(step_timing["max_batch_size"], current_batch_size)
                step_timing["max_seq_len"] = max(step_timing["max_seq_len"], current_batch_seq_len)
                step_timing["max_micro_tokens"] = max(step_timing["max_micro_tokens"], current_batch_tokens)

            is_last_micro_step = micro_step == max(int(self.config.train.grad_accum_steps), 1) - 1
            sync_context = contextlib.nullcontext() if is_last_micro_step else self.accelerator.no_sync()
            with sync_context:
                if memory_probe is not None:
                    torch.cuda.reset_peak_memory_stats(self.accelerator.device)
                if step_timing is not None:
                    sync_device_for_timing(self.accelerator.device)
                    forward_start_time = time.perf_counter()
                with self.accelerator.autocast():
                    outputs = self.model(
                        processed["text_tokens"],
                        processed["text_mask"],
                        processed["audio_feats"],
                        processed["audio_mask"],
                        processed["loss_mask"],
                        processed["labels"],
                        progress=step / max(1, self.config.train.num_iters),
                    )
                if step_timing is not None:
                    sync_device_for_timing(self.accelerator.device)
                    step_timing["forward"] += time.perf_counter() - forward_start_time
                if memory_probe is not None:
                    torch.cuda.synchronize(self.accelerator.device)
                    forward_peak = torch.cuda.max_memory_allocated(self.accelerator.device)
                    if forward_peak > memory_probe["forward_peak_allocated"]:
                        memory_probe["forward_peak_allocated"] = forward_peak
                        memory_probe["forward_peak_tokens"] = current_batch_tokens

                lambdas = dict(self.config.train.lambdas)
                total_loss = 0.0
                for key, value in outputs.items():
                    if key.startswith("loss/"):
                        total_loss = total_loss + value * lambdas.get(key, 1.0) / max(
                            int(self.config.train.grad_accum_steps),
                            1,
                        )
                        loss_dict[key] = value.detach()

                if memory_probe is not None:
                    torch.cuda.reset_peak_memory_stats(self.accelerator.device)
                if step_timing is not None:
                    sync_device_for_timing(self.accelerator.device)
                    backward_start_time = time.perf_counter()
                self.accelerator.backward(total_loss)
                if step_timing is not None:
                    sync_device_for_timing(self.accelerator.device)
                    step_timing["backward"] += time.perf_counter() - backward_start_time
                if memory_probe is not None:
                    torch.cuda.synchronize(self.accelerator.device)
                    backward_peak = torch.cuda.max_memory_allocated(self.accelerator.device)
                    if backward_peak > memory_probe["backward_peak_allocated"]:
                        memory_probe["backward_peak_allocated"] = backward_peak
                        memory_probe["backward_peak_tokens"] = current_batch_tokens

        if step_timing is not None:
            sync_device_for_timing(self.accelerator.device)
            optim_start_time = time.perf_counter()

        scaler = getattr(self.accelerator, "scaler", None)
        if scaler is not None:
            scaler.unscale_(self.optimizer)
        # Real clipping at 1.0.
        grad_norm = self.accelerator.clip_grad_norm(self.model, max_norm=1.0)
        self.accelerator.step(self.optimizer)
        self.accelerator.update()
        self.scheduler.step()

        if step_timing is not None:
            sync_device_for_timing(self.accelerator.device)
            step_timing["optim"] = time.perf_counter() - optim_start_time
            step_timing["step"] = time.perf_counter() - step_start_time
            if timing_active:
                for key in ("data", "forward", "backward", "optim", "step"):
                    self.timing_interval[key] += step_timing[key]
                self.timing_interval["tokens"] += step_timing["tokens"]
                self.timing_interval["micro_steps"] += step_timing["micro_steps"]
                self.timing_interval["steps"] += 1
            if rank_diag_active:
                update_rank_diag_interval(self.rank_diag_interval, step_timing)

        if self.accelerator.rank == 0 and not self.has_logged_cuda_memory:
            if memory_probe is not None:
                torch.cuda.synchronize(self.accelerator.device)
                memory_probe["step_end_allocated"] = torch.cuda.memory_allocated(self.accelerator.device)
                memory_probe["step_end_reserved"] = torch.cuda.memory_reserved(self.accelerator.device)
                log_cuda_peak_memory_once(
                    tracker=self.tracker,
                    probe=memory_probe,
                    device=self.accelerator.device,
                    step=step,
                )
            log_cuda_memory_breakdown_once(
                tracker=self.tracker,
                model=self.unwrapped_model,
                optimizer=self.optimizer,
                batch_tensors=processed,
                device=self.accelerator.device,
                step=step,
            )
            self.has_logged_cuda_memory = True

        if step % self.config.train.log_interval == 0 or step == self.config.train.num_iters - 1:
            loss_values = {
                key: value.item() if isinstance(value, torch.Tensor) else float(value)
                for key, value in loss_dict.items()
            }
            loss_values["lr"] = float(self.optimizer.param_groups[0]["lr"])
            sample_progress = gather_global_sample_progress(self.loop_state.local_samples_seen, self.accelerator)
            if self.accelerator.rank == 0 and sample_progress is not None:
                global_samples_seen = sample_progress["global_samples_seen"]
                loss_values["samples/seen"] = float(global_samples_seen)
                loss_values["epoch"] = float(global_samples_seen / max(1, self.num_train_samples))
            loss_values["grad_norm"] = float(grad_norm)
            if timing_active and self.timing_interval["steps"] > 0:
                interval_steps = self.timing_interval["steps"]
                interval_step_total = max(self.timing_interval["step"], 1e-12)
                loss_values["time/data_ms"] = 1000.0 * self.timing_interval["data"] / interval_steps
                loss_values["time/forward_ms"] = 1000.0 * self.timing_interval["forward"] / interval_steps
                loss_values["time/backward_ms"] = 1000.0 * self.timing_interval["backward"] / interval_steps
                loss_values["time/optim_ms"] = 1000.0 * self.timing_interval["optim"] / interval_steps
                loss_values["time/step_ms"] = 1000.0 * self.timing_interval["step"] / interval_steps
                loss_values["throughput/tokens_per_s"] = self.timing_interval["tokens"] / interval_step_total
                loss_values["throughput/micro_steps_per_s"] = (
                    self.timing_interval["micro_steps"] / interval_step_total
                )
            self.tracker.log_metrics(loss_values, split="train")
            if timing_active:
                self.timing_interval = {
                    "data": 0.0,
                    "forward": 0.0,
                    "backward": 0.0,
                    "optim": 0.0,
                    "step": 0.0,
                    "tokens": 0,
                    "micro_steps": 0,
                    "steps": 0,
                }
            if rank_diag_active and self.rank_diag_interval["steps"] > 0:
                gathered_rank_diag = gather_rank_diag_interval(self.rank_diag_interval, self.accelerator)
                if self.accelerator.rank == 0 and gathered_rank_diag:
                    self.tracker.print(format_rank_diag_summary(step, gathered_rank_diag))
                self.rank_diag_interval = new_rank_diag_interval()

        return {
            "grad_norm": grad_norm,
        }

    def run_validation_and_checkpoint(self, step: int) -> None:
        schedule = build_svs_eval_schedule(
            step=step,
            total_steps=self.config.train.num_iters,
            valid_interval=self.config.train.valid_interval,
            audio_eval_interval=self.audio_eval_interval,
            has_validation=self.val_loader is not None,
            validate_at_step_zero=self.config.train.validate_at_step_zero,
            audio_validate_at_step_zero=self.config.train.audio_validate_at_step_zero,
        )
        if schedule.run_val_loss or schedule.run_audio_eval:
            self.optimizer.zero_grad(set_to_none=True)
            # NOTE: empty_cache() disabled — same root cause as in validation.py:
            # it frees FSDP internal buffers still referenced by NCCL collectives,
            # leading to CUDA illegal memory access on the next training step.
            # torch.cuda.empty_cache()
            validate_svs(
                self.model,
                self.val_loader,
                self.accelerator,
                self.tracker,
                self.config.train.lambdas,
                writer=self.writer,
                step=step,
                val_ds=self.val_ds,
                sample_rate=self.config.train.sample_rate,
                tokenizer=self.base_tokenizer,
                valid_interval=self.config.train.valid_interval,
                run_loss_eval=schedule.run_val_loss,
                run_audio_eval=schedule.run_audio_eval,
                val_audio_samples=self.config.eval.val_audio_samples,
                val_tb_max_samples=self.config.eval.val_tb_max_samples,
                val_max_len=self.config.eval.val_max_len,
                audio_eval_batch_size=(
                    self.config.train.val_batch_size
                    if self.config.train.val_batch_size >= 0
                    else self.config.train.batch_size * 4
                ),
                evaluator=self.evaluator,
                eval_ref_cache=self.eval_ref_cache,
                val_song_index=self.val_song_index,
                prompt_max_frames=self.config.data.prompt_max_frames,
                prompt_source_ds=self.prompt_pool_ds,
                prompt_source_song_index=self.prompt_pool_song_index,
                prompt_source_val_offset=self.val_offset,
                load_audio_vae_fn=self.load_audio_vae,
            )

        save_permanent = should_save_checkpoint(
            step=step,
            save_interval=self.config.train.save_interval,
            total_steps=self.config.train.num_iters,
            skip_step_zero=True,
        )
        save_transient = (
            not save_permanent
            and self.config.train.transient_save_interval > 0
            and should_save_checkpoint(
                step=step,
                save_interval=self.config.train.transient_save_interval,
                total_steps=self.config.train.num_iters,
                skip_step_zero=True,
            )
        )
        if save_permanent or save_transient:
            self.accelerator.barrier()
            save_training_checkpoint(
                self.model,
                self.optimizer,
                self.scheduler,
                runtime=self.runtime,
                step=step,
                pretrained_path=self.config.model.pretrained_path,
                hf_model_id=self.config.model.hf_model_id,
                distribute=self.config.model.distribute,
                tokenizer=self.base_tokenizer,
                loop_state=self.loop_state,
                is_transient=save_transient,
            )
            self.accelerator.barrier()

    def run(self) -> None:
        self.build_runtime()
        self.build_model_and_tokenizer()
        self.build_datasets_and_loaders()
        self.build_eval_context()
        self.build_training_state()

        if self.accelerator.rank == 0:
            self.tracker.print(
                f"Starting SVS training from step {self.resume_context.start_step} "
                f"to {self.config.train.num_iters}..."
            )

        with self.tracker.live():
            for step in range(self.resume_context.start_step, self.config.train.num_iters):
                self.train_one_step(step)
                self.run_validation_and_checkpoint(step)

        # No post-loop final save: should_save_checkpoint() returns True at
        # step=num_iters-1, so run_validation_and_checkpoint already wrote a
        # permanent ckpt via save_training_checkpoint. A second consecutive
        # FSDP2 full_state_dict + optimizer all-gather here hangs NCCL.
        if self.accelerator.rank == 0:
            self.tracker.print("SVS training completed!")
        close_training_runtime(self.runtime)


def run(config: SVSTrainConfig) -> None:
    SVSRunner(config).run()


def train(**kwargs) -> None:
    run(load_svs_train_config(kwargs))
