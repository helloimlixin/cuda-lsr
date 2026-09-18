
#!/usr/bin/env python3
"""
Multi-GPU Muon systems benchmark on a real language model + WikiText-103.

Purpose
-------
Compare two algorithmically equivalent Muon execution schedules:

  --muon-mode serial
      Every hidden 2-D weight matrix is orthogonalized independently.

  --muon-mode batched
      Same-shaped (after orientation) momentum matrices are stacked in
      small chunks and the five Newton-Schulz iterations are evaluated
      with batched GEMMs.

Both modes use:
  * identical Muon momentum / Nesterov update,
  * identical quintic Newton-Schulz coefficients,
  * identical number of NS steps,
  * identical learning-rate adjustment,
  * AdamW for embeddings, norms, biases, LM head, and other non-Muon params.

Distributed mode
----------------
Launch with torchrun. This first experiment intentionally uses DDP, not FSDP:
each GPU owns a full model replica and DDP synchronizes gradients.

Examples
--------
# 4 GPUs, GPT-2 XL (1.5B), serial Muon:
torchrun --standalone --nproc_per_node=4 muon_lm_ddp.py \
  --model-id openai-community/gpt2-xl \
  --muon-mode serial \
  --steps 40 --warmup-steps 10 \
  --seq-len 512 --micro-batch 1 --grad-accum 2 \
  --output gpt2xl_serial.json

# Same run, shape-batched Muon:
torchrun --standalone --nproc_per_node=4 muon_lm_ddp.py \
  --model-id openai-community/gpt2-xl \
  --muon-mode batched \
  --muon-batch-matrices 8 \
  --steps 40 --warmup-steps 10 \
  --seq-len 512 --micro-batch 1 --grad-accum 2 \
  --output gpt2xl_batched.json

# 8 GPUs:
torchrun --standalone --nproc_per_node=8 muon_lm_ddp.py \
  --model-id openai-community/gpt2-xl \
  --muon-mode batched --muon-batch-matrices 8 \
  --steps 40 --warmup-steps 10 \
  --seq-len 512 --micro-batch 1 --grad-accum 1 \
  --output gpt2xl_8gpu_batched.json

# Larger model, if one full replica fits on each GPU:
torchrun --standalone --nproc_per_node=8 muon_lm_ddp.py \
  --model-id EleutherAI/gpt-neo-2.7B \
  --muon-mode batched --muon-batch-matrices 4 \
  --steps 30 --warmup-steps 8 \
  --seq-len 512 --micro-batch 1 --grad-accum 1 \
  --output gptneo27b_8gpu_batched.json

Notes
-----
* This is a systems experiment first. Use the same seed/data/LRs for serial
  and batched runs before interpreting loss differences.
* DDP does NOT shard model memory. If a model does not fit per GPU, use FSDP.
* The script caches tokenized WikiText-103 locally so data loading is not in
  the timed training loop.
"""

import argparse
import contextlib
import json
import math
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def init_distributed():
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size


def barrier():
    if dist.is_initialized():
        dist.barrier()


def rank0_print(rank, *args, **kwargs):
    if rank == 0:
        print(*args, **kwargs, flush=True)


def max_across_ranks(x, device):
    t = torch.tensor(float(x), device=device)
    if dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def mean_across_ranks(x, device):
    t = torch.tensor(float(x), device=device)
    if dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= dist.get_world_size()
    return float(t.item())


# ---------------------------------------------------------------------------
# Dataset cache
# ---------------------------------------------------------------------------

def require_hf():
    try:
        from datasets import load_dataset  # noqa: F401
        from transformers import AutoTokenizer, AutoModelForCausalLM  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Install dependencies first:\n"
            "  pip install -U transformers datasets accelerate"
        ) from exc


def build_token_cache(
    tokenizer,
    path,
    split,
    max_tokens,
    rank,
):
    """
    Build one flat int32 token stream on rank 0, then all ranks load it.
    """
    path = Path(path)

    if rank == 0 and not path.exists():
        from datasets import load_dataset

        print(f"[rank0] building token cache: {path}", flush=True)

        ds = load_dataset(
            "Salesforce/wikitext",
            "wikitext-103-raw-v1",
            split=split,
        )

        pieces = []
        total = 0
        eos = tokenizer.eos_token_id

        batch_text = []
        batch_chars = 0

        def flush_texts(texts):
            if not texts:
                return []
            enc = tokenizer(
                texts,
                add_special_tokens=False,
                padding=False,
                truncation=False,
            )["input_ids"]

            out = []
            for ids in enc:
                if ids:
                    out.append(
                        torch.tensor(
                            ids + [eos],
                            dtype=torch.int32,
                        )
                    )
            return out

        for row in ds:
            text = row["text"]
            if not text:
                continue

            batch_text.append(text)
            batch_chars += len(text)

            if len(batch_text) >= 256 or batch_chars >= 1_000_000:
                new = flush_texts(batch_text)
                pieces.extend(new)
                total += sum(x.numel() for x in new)
                batch_text = []
                batch_chars = 0

                if total >= max_tokens:
                    break

        if total < max_tokens and batch_text:
            new = flush_texts(batch_text)
            pieces.extend(new)
            total += sum(x.numel() for x in new)

        tokens = torch.cat(pieces)[:max_tokens].contiguous()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tokens, path)

        print(
            f"[rank0] cached {tokens.numel():,} tokens -> {path}",
            flush=True,
        )

    barrier()

    if not path.exists():
        raise FileNotFoundError(path)

    tokens = torch.load(path, map_location="cpu")

    if tokens.dtype != torch.int32:
        tokens = tokens.to(torch.int32)

    return tokens.contiguous()


def deterministic_batch(
    tokens,
    optimizer_step,
    micro_step,
    rank,
    world_size,
    micro_batch,
    seq_len,
    grad_accum,
    device,
):
    """
    Deterministic sequential blocks. Serial and batched optimizer runs with
    the same command-line settings see exactly the same token sequence.
    """
    block = seq_len
    nblocks = tokens.numel() // block

    logical_micro = (
        optimizer_step * grad_accum
        + micro_step
    )

    first = (
        logical_micro * world_size * micro_batch
        + rank * micro_batch
    )

    rows = []

    for b in range(micro_batch):
        block_id = (first + b) % nblocks
        start = block_id * block

        rows.append(
            tokens[
                start : start + block
            ].to(torch.long)
        )

    x = torch.stack(rows, dim=0)

    return x.to(
        device=device,
        non_blocking=True,
    )


# ---------------------------------------------------------------------------
# Muon math
# ---------------------------------------------------------------------------

NS_A = 3.4445
NS_B = -4.7750
NS_C = 2.0315


def adjusted_muon_lr(
    lr,
    shape,
    mode,
):
    rows, cols = shape

    if mode == "original":
        ratio = math.sqrt(
            max(1.0, rows / cols)
        )

    elif mode == "match_rms_adamw":
        ratio = (
            0.2
            * math.sqrt(
                max(rows, cols)
            )
        )

    elif mode == "none":
        ratio = 1.0

    else:
        raise ValueError(mode)

    return lr * ratio


@torch.no_grad()
def zeropower_serial(
    G,
    ns_steps,
    eps,
):
    """
    Same quintic Newton-Schulz form used by current Muon implementations.
    """
    X = G.to(
        dtype=torch.bfloat16,
        copy=True,
    )

    transposed = (
        X.shape[0] > X.shape[1]
    )

    if transposed:
        X = X.T

    X.div_(
        X.norm().clamp(min=eps)
    )

    for _ in range(ns_steps):
        A = X @ X.T
        B = (
            NS_B * A
            + NS_C * (A @ A)
        )
        X = (
            NS_A * X
            + B @ X
        )

    if transposed:
        X = X.T

    return X


@torch.no_grad()
def zeropower_batched_oriented(
    X,
    ns_steps,
    eps,
):
    """
    X is already oriented so:
        X.shape[-2] <= X.shape[-1]

    X shape:
        [B, m, n]

    Same polynomial as serial Muon, evaluated with bmm.
    """
    X = X.to(
        dtype=torch.bfloat16,
        copy=True,
    )

    norms = (
        X.flatten(1)
        .norm(dim=1)
        .clamp(min=eps)
    )

    X.div_(
        norms[:, None, None]
    )

    XT = X.transpose(-1, -2)

    for _ in range(ns_steps):
        A = torch.bmm(
            X,
            X.transpose(-1, -2),
        )

        A2 = torch.bmm(
            A,
            A,
        )

        B = (
            NS_B * A
            + NS_C * A2
        )

        X = (
            NS_A * X
            + torch.bmm(B, X)
        )

    return X


# ---------------------------------------------------------------------------
# Hybrid Muon + AdamW optimizer
# ---------------------------------------------------------------------------

class MuonAdamW:
    def __init__(
        self,
        named_parameters,
        muon_mode,
        muon_lr,
        muon_momentum,
        muon_weight_decay,
        muon_nesterov,
        muon_ns_steps,
        muon_eps,
        muon_lr_adjust,
        muon_batch_matrices,
        adam_lr,
        adam_betas,
        adam_weight_decay,
        rank,
    ):
        self.muon_mode = muon_mode
        self.muon_lr = muon_lr
        self.muon_momentum = muon_momentum
        self.muon_weight_decay = muon_weight_decay
        self.muon_nesterov = muon_nesterov
        self.muon_ns_steps = muon_ns_steps
        self.muon_eps = muon_eps
        self.muon_lr_adjust = muon_lr_adjust
        self.muon_batch_matrices = muon_batch_matrices
        self.rank = rank

        named_parameters = [
            (n, p)
            for n, p in named_parameters
            if p.requires_grad
        ]

        # Input/output embeddings should use AdamW.
        embed_like_names = (
            "wte",
            "wpe",
            "embed",
            "embedding",
            "lm_head",
            "output_projection",
            "word_embeddings",
            "position_embeddings",
        )

        self.muon_named = []
        self.adam_named = []

        seen = set()

        for name, p in named_parameters:
            # Protect against tied weights appearing under multiple names.
            if id(p) in seen:
                continue
            seen.add(id(p))

            excluded = any(
                key in name.lower()
                for key in embed_like_names
            )

            if p.ndim == 2 and not excluded:
                self.muon_named.append(
                    (name, p)
                )
            else:
                self.adam_named.append(
                    (name, p)
                )

        self.momentum = {
            p: torch.zeros_like(
                p,
                memory_format=torch.preserve_format,
            )
            for _, p in self.muon_named
        }

        adam_params = [
            p for _, p in self.adam_named
        ]

        self.adam = torch.optim.AdamW(
            adam_params,
            lr=adam_lr,
            betas=adam_betas,
            weight_decay=adam_weight_decay,
            fused=True,
        ) if adam_params else None

        # Group Muon params by oriented matrix shape.
        self.shape_groups = defaultdict(list)

        for name, p in self.muon_named:
            r, c = p.shape

            if r <= c:
                oriented = (r, c)
                transpose = False
            else:
                oriented = (c, r)
                transpose = True

            self.shape_groups[
                oriented
            ].append(
                (name, p, transpose)
            )

    def zero_grad(self):
        for _, p in self.muon_named:
            p.grad = None

        if self.adam is not None:
            self.adam.zero_grad(
                set_to_none=True
            )

    @torch.no_grad()
    def _momentum_update(self, p):
        g = p.grad
        buf = self.momentum[p]

        buf.mul_(
            self.muon_momentum
        ).add_(g)

        if self.muon_nesterov:
            return (
                g
                + self.muon_momentum * buf
            )

        return buf

    @torch.no_grad()
    def _apply_update(
        self,
        p,
        update,
    ):
        lr_adj = adjusted_muon_lr(
            self.muon_lr,
            p.shape,
            self.muon_lr_adjust,
        )

        if self.muon_weight_decay != 0.0:
            # Matches current Muon convention: WD uses base LR.
            p.mul_(
                1.0
                - self.muon_lr
                * self.muon_weight_decay
            )

        p.add_(
            update.to(dtype=p.dtype),
            alpha=-lr_adj,
        )

    @torch.no_grad()
    def muon_step_serial(self):
        for _, p in self.muon_named:
            if p.grad is None:
                continue

            u = self._momentum_update(p)

            ortho = zeropower_serial(
                u,
                ns_steps=self.muon_ns_steps,
                eps=self.muon_eps,
            )

            self._apply_update(
                p,
                ortho,
            )

    @torch.no_grad()
    def muon_step_batched(self):
        max_batch = (
            self.muon_batch_matrices
        )

        for oriented_shape, items in self.shape_groups.items():
            active = [
                item
                for item in items
                if item[1].grad is not None
            ]

            if not active:
                continue

            for start in range(
                0,
                len(active),
                max_batch,
            ):
                chunk = active[
                    start : start + max_batch
                ]

                matrices = []
                metadata = []

                for name, p, transpose in chunk:
                    u = self._momentum_update(
                        p
                    )

                    if transpose:
                        u = u.T

                    matrices.append(
                        u
                    )

                    metadata.append(
                        (p, transpose)
                    )

                X = torch.stack(
                    matrices,
                    dim=0,
                )

                Z = zeropower_batched_oriented(
                    X,
                    ns_steps=self.muon_ns_steps,
                    eps=self.muon_eps,
                )

                for i, (
                    p,
                    transpose,
                ) in enumerate(metadata):
                    update = Z[i]

                    if transpose:
                        update = update.T

                    self._apply_update(
                        p,
                        update,
                    )

                del X, Z, matrices

    @torch.no_grad()
    def muon_step(self):
        if self.muon_mode == "serial":
            self.muon_step_serial()

        elif self.muon_mode == "batched":
            self.muon_step_batched()

        else:
            raise ValueError(
                self.muon_mode
            )

    def adam_step(self):
        if self.adam is not None:
            self.adam.step()

    def summary(self):
        muon_params = sum(
            p.numel()
            for _, p in self.muon_named
        )

        adam_params = sum(
            p.numel()
            for _, p in self.adam_named
        )

        groups = []

        for shape, items in sorted(
            self.shape_groups.items()
        ):
            groups.append(
                {
                    "oriented_shape":
                        list(shape),
                    "count":
                        len(items),
                    "matrix_million_elements_each":
                        shape[0]
                        * shape[1]
                        / 1e6,
                    "example_names":
                        [
                            x[0]
                            for x in items[:4]
                        ],
                }
            )

        return {
            "muon_param_count":
                muon_params,
            "adam_param_count":
                adam_params,
            "muon_matrix_count":
                len(self.muon_named),
            "shape_groups":
                groups,
        }


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(
    model_id,
    init_mode,
    device,
    gradient_checkpointing,
):
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if init_mode == "pretrained":
        kwargs = {
            "low_cpu_mem_usage": True,
        }

        # Transformers versions have used both names.
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=torch.bfloat16,
                **kwargs,
            )
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                **kwargs,
            )

    elif init_mode == "random":
        config = AutoConfig.from_pretrained(
            model_id
        )

        model = AutoModelForCausalLM.from_config(
            config
        )

        model = model.to(
            dtype=torch.bfloat16
        )

    else:
        raise ValueError(init_mode)

    model.config.use_cache = False

    if gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={
                    "use_reentrant": False
                }
            )
        except TypeError:
            model.gradient_checkpointing_enable()

    model.to(device)
    model.train()

    return model, tokenizer


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    ddp_model,
    tokens,
    args,
    rank,
    world_size,
    device,
):
    ddp_model.eval()

    losses = []

    for j in range(args.eval_batches):
        x = deterministic_batch(
            tokens=tokens,
            optimizer_step=10_000_000 + j,
            micro_step=0,
            rank=rank,
            world_size=world_size,
            micro_batch=args.micro_batch,
            seq_len=args.seq_len,
            grad_accum=1,
            device=device,
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):
            out = ddp_model(
                input_ids=x,
                labels=x,
                use_cache=False,
            )

        losses.append(
            float(out.loss.detach())
        )

    local = sum(losses) / len(losses)
    mean = mean_across_ranks(
        local,
        device,
    )

    ddp_model.train()
    return mean


# ---------------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-id",
        type=str,
        default="openai-community/gpt2-xl",
    )

    parser.add_argument(
        "--init",
        choices=["pretrained", "random"],
        default="pretrained",
    )

    parser.add_argument(
        "--muon-mode",
        choices=["serial", "batched"],
        required=True,
    )

    parser.add_argument(
        "--muon-batch-matrices",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--muon-lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--muon-momentum",
        type=float,
        default=0.95,
    )

    parser.add_argument(
        "--muon-weight-decay",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--muon-ns-steps",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--muon-eps",
        type=float,
        default=1e-7,
    )

    parser.add_argument(
        "--muon-lr-adjust",
        choices=[
            "original",
            "match_rms_adamw",
            "none",
        ],
        default="original",
    )

    parser.add_argument(
        "--adam-lr",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--adam-weight-decay",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--adam-beta1",
        type=float,
        default=0.9,
    )

    parser.add_argument(
        "--adam-beta2",
        type=float,
        default=0.95,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=40,
    )

    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--seq-len",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--micro-batch",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--grad-accum",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=17,
    )

    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--train-token-cache",
        type=str,
        default="./cache/wikitext103_train_5m.pt",
    )

    parser.add_argument(
        "--val-token-cache",
        type=str,
        default="./cache/wikitext103_val_1m.pt",
    )

    parser.add_argument(
        "--train-cache-tokens",
        type=int,
        default=5_000_000,
    )

    parser.add_argument(
        "--val-cache-tokens",
        type=int,
        default=1_000_000,
    )

    parser.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help="0 disables periodic validation",
    )

    parser.add_argument(
        "--eval-batches",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--ddp-bucket-cap-mb",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--output",
        type=str,
        default="muon_run.json",
    )

    args = parser.parse_args()

    if args.muon_batch_matrices < 1:
        raise ValueError(
            "--muon-batch-matrices must be >= 1"
        )

    require_hf()

    rank, local_rank, world_size = (
        init_distributed()
    )

    device = torch.device(
        "cuda",
        local_rank,
    )

    torch.manual_seed(
        args.seed
    )
    torch.cuda.manual_seed_all(
        args.seed
    )

    torch.set_float32_matmul_precision(
        "highest"
    )

    # Keep Muon math reproducible across serial/batched runs.
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
    except Exception:
        pass

    rank0_print(
        rank,
        "=" * 100,
    )

    rank0_print(
        rank,
        "MULTI-GPU MUON LANGUAGE-MODEL BENCHMARK",
    )

    rank0_print(
        rank,
        "=" * 100,
    )

    rank0_print(
        rank,
        f"model               : {args.model_id}",
    )

    rank0_print(
        rank,
        f"init                : {args.init}",
    )

    rank0_print(
        rank,
        f"world size          : {world_size}",
    )

    rank0_print(
        rank,
        f"muon mode           : {args.muon_mode}",
    )

    rank0_print(
        rank,
        f"muon batch matrices : {args.muon_batch_matrices}",
    )

    rank0_print(
        rank,
        f"seq len             : {args.seq_len}",
    )

    rank0_print(
        rank,
        f"micro batch / rank  : {args.micro_batch}",
    )

    rank0_print(
        rank,
        f"grad accumulation   : {args.grad_accum}",
    )

    rank0_print(
        rank,
        f"GPU                 : {torch.cuda.get_device_name(local_rank)}",
    )

    # --------------------------------------------------------
    # Load model/tokenizer.
    # --------------------------------------------------------
    model, tokenizer = (
        load_model_and_tokenizer(
            model_id=args.model_id,
            init_mode=args.init,
            device=device,
            gradient_checkpointing=args.gradient_checkpointing,
        )
    )

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    rank0_print(
        rank,
        f"model params        : {total_params/1e9:.3f} B",
    )

    # --------------------------------------------------------
    # Token cache before timed training.
    # --------------------------------------------------------
    train_tokens = build_token_cache(
        tokenizer=tokenizer,
        path=args.train_token_cache,
        split="train",
        max_tokens=args.train_cache_tokens,
        rank=rank,
    )

    val_tokens = None

    if args.eval_every > 0:
        val_tokens = build_token_cache(
            tokenizer=tokenizer,
            path=args.val_token_cache,
            split="validation",
            max_tokens=args.val_cache_tokens,
            rank=rank,
        )

    barrier()

    # --------------------------------------------------------
    # Optimizer before DDP: same Parameter objects survive DDP wrapping.
    # --------------------------------------------------------
    optimizer = MuonAdamW(
        named_parameters=list(
            model.named_parameters()
        ),
        muon_mode=args.muon_mode,
        muon_lr=args.muon_lr,
        muon_momentum=args.muon_momentum,
        muon_weight_decay=args.muon_weight_decay,
        muon_nesterov=True,
        muon_ns_steps=args.muon_ns_steps,
        muon_eps=args.muon_eps,
        muon_lr_adjust=args.muon_lr_adjust,
        muon_batch_matrices=args.muon_batch_matrices,
        adam_lr=args.adam_lr,
        adam_betas=(
            args.adam_beta1,
            args.adam_beta2,
        ),
        adam_weight_decay=args.adam_weight_decay,
        rank=rank,
    )

    opt_summary = optimizer.summary()

    if rank == 0:
        print(
            f"Muon params         : "
            f"{opt_summary['muon_param_count']/1e9:.3f} B"
        )

        print(
            f"AdamW params        : "
            f"{opt_summary['adam_param_count']/1e9:.3f} B"
        )

        print(
            f"Muon matrices       : "
            f"{opt_summary['muon_matrix_count']}"
        )

        print("Muon oriented-shape groups:")

        for g in opt_summary[
            "shape_groups"
        ]:
            print(
                f"  {tuple(g['oriented_shape'])}: "
                f"{g['count']} matrices, "
                f"{g['matrix_million_elements_each']:.3f}M elems/matrix"
            )

    # --------------------------------------------------------
    # DDP wrapper.
    # --------------------------------------------------------
    if world_size > 1:
        ddp_model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
            bucket_cap_mb=args.ddp_bucket_cap_mb,
            find_unused_parameters=False,
        )
    else:
        ddp_model = model

    global_tokens_per_step = (
        world_size
        * args.micro_batch
        * args.seq_len
        * args.grad_accum
    )

    rank0_print(
        rank,
        f"global tokens/step  : {global_tokens_per_step:,}",
    )

    # --------------------------------------------------------
    # Timed training.
    # --------------------------------------------------------
    step_ms_values = []
    muon_ms_values = []
    adam_ms_values = []
    optimizer_ms_values = []
    losses = []

    peak_reset_done = False

    optimizer.zero_grad()

    for step in range(args.steps):
        # CUDA events measure GPU critical-path durations locally.
        step_start = torch.cuda.Event(
            enable_timing=True
        )
        before_muon = torch.cuda.Event(
            enable_timing=True
        )
        after_muon = torch.cuda.Event(
            enable_timing=True
        )
        after_adam = torch.cuda.Event(
            enable_timing=True
        )

        step_start.record()

        total_loss = 0.0

        for micro in range(
            args.grad_accum
        ):
            x = deterministic_batch(
                tokens=train_tokens,
                optimizer_step=step,
                micro_step=micro,
                rank=rank,
                world_size=world_size,
                micro_batch=args.micro_batch,
                seq_len=args.seq_len,
                grad_accum=args.grad_accum,
                device=device,
            )

            sync_ctx = (
                contextlib.nullcontext()
                if (
                    world_size == 1
                    or micro
                    == args.grad_accum - 1
                )
                else ddp_model.no_sync()
            )

            with sync_ctx:
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                ):
                    out = ddp_model(
                        input_ids=x,
                        labels=x,
                        use_cache=False,
                    )

                    loss = (
                        out.loss
                        / args.grad_accum
                    )

                loss.backward()

            total_loss += float(
                loss.detach()
            )

        before_muon.record()

        optimizer.muon_step()

        after_muon.record()

        optimizer.adam_step()

        after_adam.record()

        optimizer.zero_grad()

        after_adam.synchronize()

        local_step_ms = (
            step_start.elapsed_time(
                after_adam
            )
        )

        local_muon_ms = (
            before_muon.elapsed_time(
                after_muon
            )
        )

        local_adam_ms = (
            after_muon.elapsed_time(
                after_adam
            )
        )

        local_opt_ms = (
            before_muon.elapsed_time(
                after_adam
            )
        )

        # Critical rank matters for DDP throughput.
        step_ms = max_across_ranks(
            local_step_ms,
            device,
        )

        muon_ms = max_across_ranks(
            local_muon_ms,
            device,
        )

        adam_ms = max_across_ranks(
            local_adam_ms,
            device,
        )

        opt_ms = max_across_ranks(
            local_opt_ms,
            device,
        )

        loss_mean = mean_across_ranks(
            total_loss,
            device,
        )

        measured = (
            step >= args.warmup_steps
        )

        if measured:
            step_ms_values.append(
                step_ms
            )

            muon_ms_values.append(
                muon_ms
            )

            adam_ms_values.append(
                adam_ms
            )

            optimizer_ms_values.append(
                opt_ms
            )

            losses.append(
                loss_mean
            )

            if not peak_reset_done:
                torch.cuda.reset_peak_memory_stats(
                    device
                )
                peak_reset_done = True

        toks_per_s = (
            global_tokens_per_step
            / (step_ms / 1000.0)
        )

        if (
            rank == 0
            and (
                step % args.log_every == 0
                or step == args.steps - 1
            )
        ):
            tag = (
                "MEASURE"
                if measured
                else "WARMUP"
            )

            print(
                f"[{tag}] "
                f"step={step:03d} "
                f"loss={loss_mean:.5f} "
                f"step={step_ms:.1f}ms "
                f"opt={opt_ms:.1f}ms "
                f"muon={muon_ms:.1f}ms "
                f"adam={adam_ms:.1f}ms "
                f"tok/s={toks_per_s:,.0f}",
                flush=True,
            )

        if (
            args.eval_every > 0
            and (step + 1)
            % args.eval_every == 0
        ):
            val_loss = evaluate(
                ddp_model=ddp_model,
                tokens=val_tokens,
                args=args,
                rank=rank,
                world_size=world_size,
                device=device,
            )

            rank0_print(
                rank,
                f"[eval] step={step+1} "
                f"val_loss={val_loss:.6f}",
            )

    # --------------------------------------------------------
    # Summary.
    # --------------------------------------------------------
    if not step_ms_values:
        raise RuntimeError(
            "No measured steps. "
            "Set --steps > --warmup-steps."
        )

    peak_gb_local = (
        torch.cuda.max_memory_allocated(
            device
        )
        / 1024**3
    )

    peak_gb = max_across_ranks(
        peak_gb_local,
        device,
    )

    summary = {
        "model_id":
            args.model_id,

        "init":
            args.init,

        "world_size":
            world_size,

        "gpu":
            torch.cuda.get_device_name(
                local_rank
            ),

        "model_params":
            total_params,

        "muon_mode":
            args.muon_mode,

        "muon_batch_matrices":
            args.muon_batch_matrices,

        "muon_lr":
            args.muon_lr,

        "muon_lr_adjust":
            args.muon_lr_adjust,

        "muon_ns_steps":
            args.muon_ns_steps,

        "adam_lr":
            args.adam_lr,

        "seq_len":
            args.seq_len,

        "micro_batch":
            args.micro_batch,

        "grad_accum":
            args.grad_accum,

        "global_tokens_per_step":
            global_tokens_per_step,

        "steps":
            args.steps,

        "warmup_steps":
            args.warmup_steps,

        "median_step_ms":
            statistics.median(
                step_ms_values
            ),

        "median_optimizer_ms":
            statistics.median(
                optimizer_ms_values
            ),

        "median_muon_ms":
            statistics.median(
                muon_ms_values
            ),

        "median_adam_ms":
            statistics.median(
                adam_ms_values
            ),

        "median_tokens_per_second":
            global_tokens_per_step
            / (
                statistics.median(
                    step_ms_values
                )
                / 1000.0
            ),

        "mean_measured_loss":
            sum(losses)
            / len(losses),

        "final_measured_loss":
            losses[-1],

        "peak_allocated_gb":
            peak_gb,

        "optimizer_structure":
            opt_summary,
    }

    if rank == 0:
        print()
        print("=" * 100)
        print("SUMMARY")
        print("=" * 100)

        for key in [
            "median_step_ms",
            "median_optimizer_ms",
            "median_muon_ms",
            "median_adam_ms",
            "median_tokens_per_second",
            "mean_measured_loss",
            "final_measured_loss",
            "peak_allocated_gb",
        ]:
            print(
                f"{key:28s}: "
                f"{summary[key]}"
            )

        output = Path(
            args.output
        )

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output.open(
            "w"
        ) as f:
            json.dump(
                summary,
                f,
                indent=2,
            )

        print(
            f"wrote: {output}",
            flush=True,
        )

    barrier()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

