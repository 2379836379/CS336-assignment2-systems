from __future__ import annotations

import argparse
import statistics
import timeit
from dataclasses import dataclass

import torch

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW


@dataclass(frozen=True)
class ModelConfig:
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int


MODEL_SIZES: dict[str, ModelConfig] = {
    "small": ModelConfig(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": ModelConfig(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": ModelConfig(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": ModelConfig(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
    "10b": ModelConfig(d_model=4608, d_ff=12288, num_layers=50, num_heads=36),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark a CS336 Transformer training or inference step.")
    parser.add_argument("--model-size", choices=sorted(MODEL_SIZES), default="small")
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=10)
    parser.add_argument("--mode", choices=("forward", "forward-backward", "train"), default="train")
    parser.add_argument("--device", default="cuda", help="Torch device string, e.g. cuda, cuda:0, cpu.")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def parse_dtype(dtype_name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_name]


def build_model(args: argparse.Namespace) -> BasicsTransformerLM:
    cfg = MODEL_SIZES[args.model_size]
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=cfg.d_model,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
    )
    return model.to(device=args.device, dtype=parse_dtype(args.dtype))


def make_batch(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor]:
    shape = (args.batch_size, args.context_length)
    inputs = torch.randint(args.vocab_size, shape, device=args.device)
    targets = torch.randint(args.vocab_size, shape, device=args.device)
    return inputs, targets


def synchronize_if_needed(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def benchmark_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    mode: str,
) -> None:
    if mode == "forward":
        with torch.no_grad():
            _ = model(inputs)
        return

    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = cross_entropy(logits, targets)
    loss.backward()

    if mode == "train":
        optimizer.step()


def run_benchmark(args: argparse.Namespace) -> list[float]:
    model = build_model(args)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    inputs, targets = make_batch(args)

    for _ in range(args.warmup_steps):
        benchmark_step(model, optimizer, inputs, targets, args.mode)
        synchronize_if_needed(args.device)

    measurements_ms: list[float] = []
    timer = timeit.default_timer
    for _ in range(args.measure_steps):
        start = timer()
        benchmark_step(model, optimizer, inputs, targets, args.mode)
        synchronize_if_needed(args.device)
        end = timer()
        measurements_ms.append((end - start) * 1000.0)

    return measurements_ms


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested, but torch.cuda.is_available() is False.")

    measurements_ms = run_benchmark(args)
    mean_ms = statistics.mean(measurements_ms)
    std_ms = statistics.stdev(measurements_ms) if len(measurements_ms) > 1 else 0.0

    print(f"mode={args.mode}")
    print(f"model_size={args.model_size}")
    print(f"device={args.device}")
    print(f"dtype={args.dtype}")
    print(f"batch_size={args.batch_size}")
    print(f"context_length={args.context_length}")
    print(f"warmup_steps={args.warmup_steps}")
    print(f"measure_steps={args.measure_steps}")
    print("measurements_ms=" + ",".join(f"{value:.3f}" for value in measurements_ms))
    print(f"mean_ms={mean_ms:.3f}")
    print(f"std_ms={std_ms:.3f}")


if __name__ == "__main__":
    main()
