#!/usr/bin/env python3
"""
Real multi-GPU GPT training with Meta's official Distributed Shampoo / SOAP.

Purpose
-------
Before patching the production optimizer with our batched matrix-function path,
measure the REAL optimizer workload on GPT-2 XL (or another HF causal LM):

  * normal optimizer-step latency
  * periodic preconditioner-update latency
  * end-to-end step time / tokens per second
  * loss trajectory
  * peak memory

Modes
-----
  --optimizer shampoo
      Official DistributedShampoo with AdamW grafting.

  --optimizer soap
      Official eigenvalue-corrected Shampoo / SOAP-style preconditioner.

This script intentionally uses the unmodified official implementation.
That gives us the production baseline we need before substituting the
batched/hybrid matrix-function phase.

Current official Distributed Shampoo supports DDP directly. For model sizes
that do not fit as full replicas, move to FSDP in the next script.

Install
-------
git clone https://github.com/facebookresearch/optimizers.git
cd optimizers
pip install .
cd -

pip install -U transformers datasets accelerate

Examples
--------
# Shampoo, 4 GPUs, GPT-2 XL
torchrun --standalone --nproc_per_node=4 shampoo_soap_lm_ddp.py \
  --model-id openai-community/gpt2-xl \
  --optimizer shampoo \
  --steps 40 --warmup-steps 10 \
  --seq-len 512 --micro-batch 1 --grad-accum 2 \
  --max-preconditioner-dim 512 \
  --precondition-frequency 5 \
  --output gpt2xl_shampoo_4gpu.json

# SOAP / eigenvalue-corrected Shampoo
torchrun --standalone --nproc_per_node=4 shampoo_soap_lm_ddp.py \
  --model-id openai-community/gpt2-xl \
  --optimizer soap \
  --steps 40 --warmup-steps 10 \
  --seq-len 512 --micro-batch 1 --grad-accum 2 \
  --max-preconditioner-dim 512 \
  --precondition-frequency 5 \
  --output gpt2xl_soap_4gpu.json

Notes
-----
* The first periodic preconditioner step can include initialization/compile
  effects. We report all periodic steps individually and their median.
* The token stream is deterministic across runs with identical settings.
* The optimizer is official/unmodified. This is a workload verification run,
  not yet our optimized batched-root implementation.
"""

import argparse
import contextlib
import inspect
import json
import os
import statistics
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


# ============================================================================
# Distributed helpers
# ============================================================================

def init_dist():
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        torch.cuda.set_device(0)
    return rank, local_rank, world_size


def barrier():
    if dist.is_initialized():
        dist.barrier()


def reduce_max(value, device):
    t = torch.tensor(float(value), device=device)
    if dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def reduce_mean(value, device):
    t = torch.tensor(float(value), device=device)
    if dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= dist.get_world_size()
    return float(t.item())


def r0(rank, *args):
    if rank == 0:
        print(*args, flush=True)


# ============================================================================
# Dependencies / official optimizer
# ============================================================================

def import_dependencies():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError(
            "Install HF dependencies:\n"
            "  pip install -U transformers datasets accelerate"
        ) from e

    try:
        import distributed_shampoo
        from distributed_shampoo import (
            AdamPreconditionerConfig,
            DistributedShampoo,
            WeightDecayType,
        )
    except ImportError as e:
        raise RuntimeError(
            "Meta Distributed Shampoo is not installed.\n\n"
            "Install the current official implementation:\n"
            "  git clone https://github.com/facebookresearch/optimizers.git\n"
            "  cd optimizers\n"
            "  pip install .\n"
        ) from e

    return (
        AutoModelForCausalLM,
        AutoTokenizer,
        load_dataset,
        distributed_shampoo,
        AdamPreconditionerConfig,
        DistributedShampoo,
        WeightDecayType,
    )


# ============================================================================
# Token cache
# ============================================================================

def build_token_cache(
    tokenizer,
    load_dataset,
    cache_path,
    split,
    max_tokens,
    rank,
):
    cache_path = Path(cache_path)

    if rank == 0 and not cache_path.exists():
        print(f"[rank0] creating {cache_path}", flush=True)

        ds = load_dataset(
            "Salesforce/wikitext",
            "wikitext-103-raw-v1",
            split=split,
        )

        eos = tokenizer.eos_token_id
        pieces = []
        total = 0
        text_batch = []

        def flush(texts):
            if not texts:
                return []
            ids_batch = tokenizer(
                texts,
                add_special_tokens=False,
                truncation=False,
                padding=False,
            )["input_ids"]
            return [
                torch.tensor(ids + [eos], dtype=torch.int32)
                for ids in ids_batch
                if ids
            ]

        for row in ds:
            text = row["text"]
            if not text:
                continue
            text_batch.append(text)

            if len(text_batch) >= 256:
                new = flush(text_batch)
                pieces.extend(new)
                total += sum(x.numel() for x in new)
                text_batch = []

                if total >= max_tokens:
                    break

        if total < max_tokens and text_batch:
            pieces.extend(flush(text_batch))

        tokens = torch.cat(pieces)[:max_tokens].contiguous()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tokens, cache_path)
        print(f"[rank0] cached {tokens.numel():,} tokens", flush=True)

    barrier()
    return torch.load(cache_path, map_location="cpu").to(torch.int32).contiguous()


def deterministic_batch(
    tokens,
    step,
    micro,
    rank,
    world_size,
    micro_batch,
    seq_len,
    grad_accum,
    device,
):
    nblocks = tokens.numel() // seq_len
    logical_micro = step * grad_accum + micro
    first = logical_micro * world_size * micro_batch + rank * micro_batch

    rows = []
    for j in range(micro_batch):
        block_id = (first + j) % nblocks
        a = block_id * seq_len
        rows.append(tokens[a:a + seq_len].to(torch.long))

    return torch.stack(rows).to(device, non_blocking=True)


# ============================================================================
# Model
# ============================================================================

def load_model(
    AutoModelForCausalLM,
    AutoTokenizer,
    model_id,
    device,
    gradient_checkpointing,
):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )

    model.config.use_cache = False

    if gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()

    model.to(device)
    model.train()
    return model, tok


# ============================================================================
# Official Shampoo / SOAP construction
# ============================================================================

def make_optimizer(
    mode,
    model,
    lr,
    betas,
    epsilon,
    weight_decay,
    max_preconditioner_dim,
    precondition_frequency,
    start_preconditioning_step,
    AdamPreconditionerConfig,
    DistributedShampoo,
    WeightDecayType,
    distributed_shampoo_module,
    rank,
):
    common = dict(
        params=model.parameters(),
        lr=lr,
        betas=betas,
        epsilon=epsilon,
        weight_decay=weight_decay,
        max_preconditioner_dim=max_preconditioner_dim,
        precondition_frequency=precondition_frequency,
        start_preconditioning_step=start_preconditioning_step,
        weight_decay_type=WeightDecayType.DECOUPLED,
    )

    # Current production code has evolved quickly. Add optional knobs only if
    # this installed version accepts them.
    sig = inspect.signature(DistributedShampoo.__init__)

    if "eager_nan_check" in sig.parameters:
        common["eager_nan_check"] = False

    if mode == "shampoo":
        common["grafting_config"] = AdamPreconditionerConfig(
            beta2=betas[1],
            epsilon=1e-8,
        )

    elif mode == "soap":
        # Official README documents eigenvalue-corrected Shampoo as SOAP-style.
        soap_cfg = None

        # Prefer DefaultSOAPConfig when exported by the installed revision.
        if hasattr(distributed_shampoo_module, "DefaultSOAPConfig"):
            soap_cfg = getattr(
                distributed_shampoo_module,
                "DefaultSOAPConfig",
            )
            r0(rank, "SOAP config: DefaultSOAPConfig")

        elif hasattr(
            distributed_shampoo_module,
            "DefaultEigenvalueCorrectedShampooConfig",
        ):
            soap_cfg = getattr(
                distributed_shampoo_module,
                "DefaultEigenvalueCorrectedShampooConfig",
            )
            r0(rank, "SOAP config: DefaultEigenvalueCorrectedShampooConfig")

        else:
            raise RuntimeError(
                "Installed distributed_shampoo does not export "
                "DefaultSOAPConfig or DefaultEigenvalueCorrectedShampooConfig. "
                "Please update facebookresearch/optimizers."
            )

        common["preconditioner_config"] = soap_cfg

    else:
        raise ValueError(mode)

    optimizer = DistributedShampoo(**common)

    if rank == 0:
        print("DistributedShampoo signature:", sig, flush=True)

    return optimizer


# ============================================================================
# Training
# ============================================================================

def main():
    p = argparse.ArgumentParser()

    p.add_argument("--model-id", default="openai-community/gpt2-xl")
    p.add_argument("--optimizer", choices=["shampoo", "soap"], required=True)

    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--micro-batch", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=2)

    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--epsilon", type=float, default=1e-12)
    p.add_argument("--weight-decay", type=float, default=0.01)

    p.add_argument("--max-preconditioner-dim", type=int, default=512)
    p.add_argument("--precondition-frequency", type=int, default=5)
    p.add_argument("--start-preconditioning-step", type=int, default=5)

    p.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--train-cache", default="./cache/wikitext103_train_5m.pt")
    p.add_argument("--train-cache-tokens", type=int, default=5_000_000)
    p.add_argument("--output", default="shampoo_soap_run.json")

    args = p.parse_args()

    (
        AutoModelForCausalLM,
        AutoTokenizer,
        load_dataset,
        distributed_shampoo_module,
        AdamPreconditionerConfig,
        DistributedShampoo,
        WeightDecayType,
    ) = import_dependencies()

    rank, local_rank, world_size = init_dist()
    device = torch.device("cuda", local_rank)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    try:
        torch.backends.cuda.matmul.allow_tf32 = False
    except Exception:
        pass

    r0(rank, "=" * 100)
    r0(rank, "OFFICIAL DISTRIBUTED SHAMPOO / SOAP REAL-TRAINING PROFILER")
    r0(rank, "=" * 100)
    r0(rank, "model                    :", args.model_id)
    r0(rank, "optimizer                :", args.optimizer)
    r0(rank, "world size               :", world_size)
    r0(rank, "max preconditioner dim   :", args.max_preconditioner_dim)
    r0(rank, "precondition frequency   :", args.precondition_frequency)
    r0(rank, "start preconditioning    :", args.start_preconditioning_step)
    r0(rank, "GPU                      :", torch.cuda.get_device_name(local_rank))

    model, tokenizer = load_model(
        AutoModelForCausalLM,
        AutoTokenizer,
        args.model_id,
        device,
        args.gradient_checkpointing,
    )

    total_params = sum(p.numel() for p in model.parameters())
    r0(rank, f"parameters               : {total_params/1e9:.3f} B")

    train_tokens = build_token_cache(
        tokenizer,
        load_dataset,
        args.train_cache,
        "train",
        args.train_cache_tokens,
        rank,
    )

    optimizer = make_optimizer(
        mode=args.optimizer,
        model=model,
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        epsilon=args.epsilon,
        weight_decay=args.weight_decay,
        max_preconditioner_dim=args.max_preconditioner_dim,
        precondition_frequency=args.precondition_frequency,
        start_preconditioning_step=args.start_preconditioning_step,
        AdamPreconditionerConfig=AdamPreconditionerConfig,
        DistributedShampoo=DistributedShampoo,
        WeightDecayType=WeightDecayType,
        distributed_shampoo_module=distributed_shampoo_module,
        rank=rank,
    )

    if world_size > 1:
        ddp_model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
            find_unused_parameters=False,
        )
    else:
        ddp_model = model

    global_tokens = (
        world_size
        * args.micro_batch
        * args.seq_len
        * args.grad_accum
    )
    r0(rank, f"global tokens / step     : {global_tokens:,}")

    normal_opt_ms = []
    periodic_opt_ms = []
    normal_step_ms = []
    periodic_step_ms = []
    losses = []
    periodic_records = []

    optimizer.zero_grad(set_to_none=True)

    for step in range(args.steps):
        step_start = torch.cuda.Event(enable_timing=True)
        opt_start = torch.cuda.Event(enable_timing=True)
        opt_end = torch.cuda.Event(enable_timing=True)

        step_start.record()
        loss_sum = 0.0

        for micro in range(args.grad_accum):
            x = deterministic_batch(
                train_tokens,
                step,
                micro,
                rank,
                world_size,
                args.micro_batch,
                args.seq_len,
                args.grad_accum,
                device,
            )

            sync_ctx = (
                contextlib.nullcontext()
                if world_size == 1 or micro == args.grad_accum - 1
                else ddp_model.no_sync()
            )

            with sync_ctx:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = ddp_model(
                        input_ids=x,
                        labels=x,
                        use_cache=False,
                    )
                    loss = out.loss / args.grad_accum

                loss.backward()

            loss_sum += float(loss.detach())

        opt_start.record()
        optimizer.step()
        opt_end.record()
        optimizer.zero_grad(set_to_none=True)

        opt_end.synchronize()

        local_opt_ms = opt_start.elapsed_time(opt_end)
        local_step_ms = step_start.elapsed_time(opt_end)

        opt_ms = reduce_max(local_opt_ms, device)
        step_ms = reduce_max(local_step_ms, device)
        loss_mean = reduce_mean(loss_sum, device)

        # Meta uses 1-indexed optimizer steps internally. With start=frequency,
        # steps frequency-1, 2*frequency-1, ... in this zero-indexed loop are
        # expected to perform amortized preconditioner computation.
        internal_step = step + 1

        periodic = (
            internal_step >= args.start_preconditioning_step
            and (
                internal_step % args.precondition_frequency == 0
                or args.precondition_frequency == 1
            )
        )

        measured = step >= args.warmup_steps

        if measured:
            losses.append(loss_mean)

            if periodic:
                periodic_opt_ms.append(opt_ms)
                periodic_step_ms.append(step_ms)
                periodic_records.append(
                    {
                        "step": internal_step,
                        "optimizer_ms": opt_ms,
                        "step_ms": step_ms,
                        "loss": loss_mean,
                    }
                )
            else:
                normal_opt_ms.append(opt_ms)
                normal_step_ms.append(step_ms)

        tag = "PERIODIC" if periodic else "NORMAL"
        phase = "MEASURE" if measured else "WARMUP"

        if rank == 0:
            print(
                f"[{phase}/{tag}] "
                f"step={internal_step:03d} "
                f"loss={loss_mean:.5f} "
                f"step={step_ms:.1f}ms "
                f"optimizer={opt_ms:.1f}ms "
                f"tok/s={global_tokens/(step_ms/1000):,.0f}",
                flush=True,
            )

    peak_gb = reduce_max(
        torch.cuda.max_memory_allocated(device) / 1024**3,
        device,
    )

    def med(xs):
        return statistics.median(xs) if xs else None

    summary = {
        "model_id": args.model_id,
        "optimizer": args.optimizer,
        "world_size": world_size,
        "parameters": total_params,
        "max_preconditioner_dim": args.max_preconditioner_dim,
        "precondition_frequency": args.precondition_frequency,
        "start_preconditioning_step": args.start_preconditioning_step,
        "global_tokens_per_step": global_tokens,
        "median_normal_optimizer_ms": med(normal_opt_ms),
        "median_periodic_optimizer_ms": med(periodic_opt_ms),
        "median_normal_step_ms": med(normal_step_ms),
        "median_periodic_step_ms": med(periodic_step_ms),
        "periodic_over_normal_optimizer_ratio": (
            med(periodic_opt_ms) / med(normal_opt_ms)
            if normal_opt_ms and periodic_opt_ms
            else None
        ),
        "median_all_measured_step_ms": med(normal_step_ms + periodic_step_ms),
        "mean_measured_loss": sum(losses) / len(losses) if losses else None,
        "final_measured_loss": losses[-1] if losses else None,
        "peak_allocated_gb": peak_gb,
        "periodic_records": periodic_records,
    }

    if rank == 0:
        print()
        print("=" * 100)
        print("SUMMARY")
        print("=" * 100)
        for k, v in summary.items():
            if k != "periodic_records":
                print(f"{k:42s}: {v}")

        Path(args.output).write_text(json.dumps(summary, indent=2))
        print("wrote:", args.output)

    barrier()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

