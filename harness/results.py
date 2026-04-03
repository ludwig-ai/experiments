"""Results collection, persistence, and leaderboard management."""

import hashlib
import json
import os
import platform
from datetime import datetime, timezone


def get_hardware_info() -> dict:
    """Collect hardware information for result provenance."""
    info = {"cpu": platform.processor() or platform.machine(), "platform": platform.platform()}
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_memory_mb"] = torch.cuda.get_device_properties(0).total_mem // (1024 * 1024)
    except ImportError:
        pass
    try:
        import psutil

        info["ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 1)
    except ImportError:
        pass
    return info


def config_hash(config: dict) -> str:
    """Deterministic hash of a config for deduplication."""
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]


def build_result(
    dataset_name: str,
    model_name: str,
    config: dict,
    train_stats,
    eval_stats: dict,
    training_time: float,
    seed: int,
    primary_metric: str,
    primary_output: str,
) -> dict:
    """Build a standardized result dict from Ludwig training outputs."""
    import ludwig

    # Extract metrics from eval_stats
    metrics = {}
    if eval_stats and primary_output in eval_stats:
        metrics["test"] = {k: v for k, v in eval_stats[primary_output].items() if isinstance(v, (int, float))}

    # Extract validation metrics from train_stats
    if train_stats and hasattr(train_stats, "validation"):
        val_metrics = train_stats.validation.get(primary_output, {})
        val_best = {}
        for metric_name, values in val_metrics.items():
            if values:
                if "loss" in metric_name or "error" in metric_name:
                    val_best[metric_name] = min(values)
                else:
                    val_best[metric_name] = max(values)
        metrics["validation"] = val_best

    # Count epochs trained
    epochs_trained = 0
    if train_stats and hasattr(train_stats, "training"):
        combined = train_stats.training.get("combined", {})
        loss_values = combined.get("loss", [])
        epochs_trained = len(loss_values)

    return {
        "dataset": dataset_name,
        "model": model_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ludwig_version": ludwig.__version__,
        "hardware": get_hardware_info(),
        "config_hash": config_hash(config),
        "config": config,
        "metrics": metrics,
        "primary_metric": primary_metric,
        "primary_value": metrics.get("validation", {}).get(primary_metric)
        or metrics.get("test", {}).get(primary_metric),
        "training_time_seconds": round(training_time, 2),
        "epochs_trained": epochs_trained,
        "seed": seed,
    }


def save_result(result: dict, base_dir: str):
    """Save a result to the runs directory."""
    ts = result["timestamp"].replace(":", "-").replace("+", "p")
    run_dir = os.path.join(base_dir, "results", "runs", result["dataset"], result["model"])
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, f"{ts}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    return path


def load_leaderboard(base_dir: str) -> dict:
    """Load the leaderboard from disk."""
    path = os.path.join(base_dir, "results", "leaderboard.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def update_leaderboard(result: dict, base_dir: str, is_minimize: bool = False):
    """Update leaderboard if this result is the best for its dataset/model pair."""
    lb = load_leaderboard(base_dir)
    dataset = result["dataset"]
    model = result["model"]
    metric = result["primary_metric"]
    value = result["primary_value"]

    if value is None:
        return

    if dataset not in lb:
        lb[dataset] = {}

    current = lb[dataset].get(model)
    if current is None:
        is_better = True
    else:
        old_val = current.get(metric, float("inf") if is_minimize else float("-inf"))
        is_better = (value < old_val) if is_minimize else (value > old_val)

    if is_better:
        lb[dataset][model] = {
            metric: value,
            "run": result["timestamp"],
            "config_hash": result["config_hash"],
            "seed": result["seed"],
        }
        path = os.path.join(base_dir, "results", "leaderboard.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(lb, f, indent=2, sort_keys=True)


def print_leaderboard(base_dir: str):
    """Print a formatted leaderboard table."""
    lb = load_leaderboard(base_dir)
    if not lb:
        print("No results yet.")
        return

    all_models = sorted({m for ds in lb.values() for m in ds})
    header = f"{'Dataset':<30s}" + "".join(f"{m:>20s}" for m in all_models)
    print(header)
    print("-" * len(header))

    for dataset in sorted(lb):
        row = f"{dataset:<30s}"
        for model in all_models:
            entry = lb[dataset].get(model)
            if entry:
                metric_name = [k for k in entry if k not in ("run", "config_hash", "seed")][0]
                row += f"{entry[metric_name]:>20.6f}"
            else:
                row += f"{'—':>20s}"
        print(row)


def print_comparison(base_dir: str, dataset_name: str):
    """Print comparison table for all models on a single dataset."""
    lb = load_leaderboard(base_dir)
    if dataset_name not in lb:
        print(f"No results for dataset '{dataset_name}'.")
        return

    entries = lb[dataset_name]
    print(f"\n{'Model':<25s} {'Metric':<20s} {'Value':>12s} {'Seed':>6s}")
    print("-" * 65)
    for model in sorted(entries):
        entry = entries[model]
        metric_name = [k for k in entry if k not in ("run", "config_hash", "seed")][0]
        print(f"{model:<25s} {metric_name:<20s} {entry[metric_name]:>12.6f} {entry.get('seed', ''):>6}")
