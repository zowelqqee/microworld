"""Karpathy-style makemore baseline for generated name audits.

The implementation is intentionally local and compact: a character vocabulary,
fixed-length context blocks, an embedding table, one tanh hidden layer, and
softmax cross-entropy.  PyTorch is imported lazily so the main test suite can
run on machines without torch installed.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.memory_benchmark import (
    MemoryTracker,
    empty_memory_metrics,
    memory_mb_summary,
    phase_memory_metrics,
)
from core.surname_generator import load_surnames

_HERE = os.path.dirname(__file__)
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
DEFAULT_OUTPUT = os.path.join(_ROOT, "data", "generated_names_makemore_benchmark.csv")
DEFAULT_METRICS_OUTPUT = os.path.join(_ROOT, "data", "makemore_baseline_metrics.json")
AUDIT_COLUMNS = ["name", "manual_label", "notes"]


def _parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _load_clean_names(input_path: str) -> list[str]:
    return [name.strip().lower() for name in load_surnames(input_path) if name.strip()]


def has_manual_labels(path: str) -> bool:
    if not os.path.exists(path):
        return False
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "manual_label" not in reader.fieldnames:
                return False
            return any((row.get("manual_label") or "").strip() for row in reader)
    except csv.Error:
        return False


def write_makemore_audit_csv(
    names: list[str],
    output_path: str = DEFAULT_OUTPUT,
    *,
    force: bool = False,
) -> dict:
    if has_manual_labels(output_path) and not force:
        return {
            "written": False,
            "row_count": None,
            "skipped_reason": "existing manual labels present; pass --force to overwrite",
        }
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=AUDIT_COLUMNS)
        writer.writeheader()
        for name in names:
            writer.writerow({"name": name, "manual_label": "", "notes": ""})
    return {"written": True, "row_count": len(names), "skipped_reason": None}


def _unavailable_result(
    reason: str,
    output_csv: str,
    *,
    force: bool = False,
    track_memory: bool = True,
) -> dict:
    write_info = write_makemore_audit_csv([], output_csv, force=force)
    return {
        "available": False,
        "skipped_reason": reason,
        "uses_backpropagation": True,
        "uses_neural_weights": True,
        "generated_csv": output_csv,
        "generated_csv_written": write_info["written"],
        "generated_csv_write_skipped_reason": write_info["skipped_reason"],
        "memory": empty_memory_metrics(["training", "generation"], enabled=track_memory),
    }


def _build_vocab(names: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    chars = sorted(set("".join(names)))
    stoi = {ch: i + 1 for i, ch in enumerate(chars)}
    stoi["."] = 0
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def _build_dataset(names: list[str], stoi: dict[str, int], block_size: int, torch):
    xs: list[list[int]] = []
    ys: list[int] = []
    for name in names:
        context = [0] * block_size
        for ch in name + ".":
            ix = stoi[ch]
            xs.append(context)
            ys.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)


def _split_names(names: list[str], seed: int) -> tuple[list[str], list[str]]:
    shuffled = list(names)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) < 3:
        return shuffled, []
    dev_count = max(1, int(round(len(shuffled) * 0.1)))
    dev_count = min(dev_count, len(shuffled) - 1)
    return shuffled[dev_count:], shuffled[:dev_count]


def _parameter_count(model) -> int:
    return sum(int(param.numel()) for param in model.parameters())


def run_baseline(
    input_path: str,
    *,
    count: int = 100,
    seed: int = 50,
    steps: int = 50000,
    embedding_dim: int = 16,
    hidden_dim: int = 200,
    block_size: int = 3,
    batch_size: int = 32,
    learning_rate: float = 0.1,
    temperature: float = 0.8,
    max_length: int = 16,
    min_length: int = 3,
    output_csv: str = DEFAULT_OUTPUT,
    force: bool = False,
    max_attempts_per_name: int = 50,
    track_memory: bool = True,
    memory_sample_interval_ms: int = 10,
) -> dict:
    try:
        import torch
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover - depends on local environment
        return _unavailable_result(
            f"PyTorch unavailable: {exc}",
            output_csv,
            force=force,
            track_memory=track_memory,
        )

    with MemoryTracker(
        "makemore_training",
        enabled=track_memory,
        interval_ms=memory_sample_interval_ms,
    ) as training_memory_tracker:
        names = _load_clean_names(input_path)
        if not names:
            return _unavailable_result(
                "No input names available",
                output_csv,
                force=force,
                track_memory=track_memory,
            )

        train_names, dev_names = _split_names(names, seed)
        stoi, itos = _build_vocab(names)
        x_train, y_train = _build_dataset(train_names, stoi, block_size, torch)
        x_dev, y_dev = _build_dataset(dev_names, stoi, block_size, torch)
        if x_train.numel() == 0:
            return _unavailable_result(
                "No training examples available",
                output_csv,
                force=force,
                track_memory=track_memory,
            )

        class MakemoreMLP(torch.nn.Module):
            def __init__(self, vocab_size: int) -> None:
                super().__init__()
                self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
                self.hidden = torch.nn.Linear(block_size * embedding_dim, hidden_dim)
                self.output = torch.nn.Linear(hidden_dim, vocab_size)

            def forward(self, x):
                emb = self.embedding(x).view(x.shape[0], block_size * embedding_dim)
                hidden = torch.tanh(self.hidden(emb))
                return self.output(hidden)

        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed)
        model = MakemoreMLP(len(stoi))
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        def loss_for(x, y):
            if x.numel() == 0:
                return None
            with torch.no_grad():
                return float(F.cross_entropy(model(x), y).item())

        initial_loss = loss_for(x_train, y_train)
        train_start = time.perf_counter()
        loss_log: list[dict] = []
        final_loss = initial_loss
        log_interval = max(1, steps // 10) if steps else 1
        effective_batch_size = min(batch_size, x_train.shape[0])
        decay_step = int(steps * 0.7)
        for step in range(max(0, steps)):
            lr = learning_rate if step < decay_step else learning_rate * 0.1
            for group in optimizer.param_groups:
                group["lr"] = lr
            batch_ix = torch.randint(
                0,
                x_train.shape[0],
                (effective_batch_size,),
                generator=generator,
            )
            logits = model(x_train[batch_ix])
            loss = F.cross_entropy(logits, y_train[batch_ix])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.item())
            if step == 0 or (step + 1) % log_interval == 0 or step + 1 == steps:
                loss_log.append({"step": step + 1, "loss": final_loss, "learning_rate": lr})
        training_time = time.perf_counter() - train_start

        train_loss = loss_for(x_train, y_train)
        dev_loss = loss_for(x_dev, y_dev) if x_dev.numel() else None
        if train_loss is not None:
            final_loss = train_loss
    training_memory = training_memory_tracker.to_dict()

    def sample_one() -> str:
        context = [0] * block_size
        out: list[str] = []
        for _ in range(max_length):
            x = torch.tensor([context], dtype=torch.long)
            with torch.no_grad():
                logits = model(x) / max(temperature, 1e-6)
                probs = torch.softmax(logits, dim=1)
                ix = int(torch.multinomial(probs, num_samples=1, generator=generator).item())
            if ix == 0:
                break
            out.append(itos[ix])
            context = context[1:] + [ix]
        return "".join(out)

    with MemoryTracker(
        "makemore_generation",
        enabled=track_memory,
        interval_ms=memory_sample_interval_ms,
    ) as generation_memory_tracker:
        gen_start = time.perf_counter()
        generated: list[str] = []
        rejected_samples = 0
        while len(generated) < count:
            last_candidate = ""
            accepted = False
            for _ in range(max_attempts_per_name):
                candidate = sample_one()
                last_candidate = candidate
                if min_length <= len(candidate) <= max_length:
                    generated.append(candidate)
                    accepted = True
                    break
                rejected_samples += 1
            if not accepted:
                generated.append(last_candidate[:max_length])
        generation_time = time.perf_counter() - gen_start
    generation_memory = generation_memory_tracker.to_dict()

    buffer = io.BytesIO()
    torch.save(
        {
            "state_dict": model.state_dict(),
            "stoi": stoi,
            "block_size": block_size,
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
        },
        buffer,
    )
    write_info = write_makemore_audit_csv(generated, output_csv, force=force)
    params = _parameter_count(model)
    memory = {
        "available": training_memory["available"],
        "memory_metrics_available": training_memory["available"],
        "skipped_reason": training_memory["skipped_reason"],
    }
    memory.update(phase_memory_metrics("training", training_memory))
    memory.update(phase_memory_metrics("generation", generation_memory))
    memory["memory_mb"] = memory_mb_summary(memory)

    return {
        "available": True,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "train_loss": train_loss,
        "dev_loss": dev_loss,
        "loss_log": loss_log,
        "training_time_sec": training_time,
        "generation_time_sec": generation_time,
        "generated_count": len(generated),
        "rejected_samples": rejected_samples,
        "parameter_count": params,
        "trainable_parameter_count": params,
        "model_state_size_bytes": len(buffer.getvalue()),
        "uses_backpropagation": True,
        "uses_neural_weights": True,
        "generated_csv": output_csv,
        "generated_csv_written": write_info["written"],
        "generated_csv_write_skipped_reason": write_info["skipped_reason"],
        "memory": memory,
        "config": {
            "block_size": block_size,
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "steps": steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "temperature": temperature,
            "max_length": max_length,
            "min_length": min_length,
            "track_memory": track_memory,
            "memory_sample_interval_ms": memory_sample_interval_ms,
        },
    }


def write_json(result: dict, output_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train and sample a makemore-style name baseline.")
    ap.add_argument("--input", default=os.path.join(_ROOT, "data", "surnames.txt"))
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--seed", type=int, default=50)
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--hidden-dim", type=int, default=200)
    ap.add_argument("--embedding-dim", type=int, default=16)
    ap.add_argument("--block-size", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--learning-rate", type=float, default=0.1)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-length", type=int, default=16)
    ap.add_argument("--min-length", type=int, default=3)
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    ap.add_argument("--metrics-output", default=DEFAULT_METRICS_OUTPUT)
    ap.add_argument("--force", default="false")
    ap.add_argument("--track-memory", default="true")
    ap.add_argument("--memory-sample-interval-ms", type=int, default=10)
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    result = run_baseline(
        args.input,
        count=args.count,
        seed=args.seed,
        steps=args.steps,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        block_size=args.block_size,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        max_length=args.max_length,
        min_length=args.min_length,
        output_csv=args.output,
        force=_parse_bool(args.force),
        track_memory=_parse_bool(args.track_memory),
        memory_sample_interval_ms=args.memory_sample_interval_ms,
    )
    write_json(result, args.metrics_output)
    if not result["available"]:
        print(result["skipped_reason"], file=sys.stderr)
    elif not result.get("generated_csv_written", True):
        print(result["generated_csv_write_skipped_reason"], file=sys.stderr)
    print(f"Wrote makemore metrics -> {args.metrics_output}", file=sys.stderr)


if __name__ == "__main__":
    main()
