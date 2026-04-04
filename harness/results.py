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
            props = torch.cuda.get_device_properties(0)
            mem = getattr(props, "total_memory", None) or getattr(props, "total_mem", 0)
            info["gpu_memory_mb"] = mem // (1024 * 1024)
    except (ImportError, Exception):
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

    # Extract ALL metrics from eval_stats for ALL output features
    metrics = {}
    if eval_stats:
        test_metrics = {}
        for feat_name, feat_metrics in eval_stats.items():
            if isinstance(feat_metrics, dict):
                scalar_metrics = {k: v for k, v in feat_metrics.items() if isinstance(v, (int, float))}
                if scalar_metrics:
                    test_metrics[feat_name] = scalar_metrics
        metrics["test"] = test_metrics

    # Extract validation metrics from train_stats for ALL output features
    if train_stats and hasattr(train_stats, "validation"):
        val_best_all = {}
        for feat_name, feat_val_metrics in train_stats.validation.items():
            if isinstance(feat_val_metrics, dict):
                val_best = {}
                for metric_name, values in feat_val_metrics.items():
                    if values and isinstance(values, list) and isinstance(values[0], (int, float)):
                        if "loss" in metric_name or "error" in metric_name:
                            val_best[metric_name] = min(values)
                        else:
                            val_best[metric_name] = max(values)
                if val_best:
                    val_best_all[feat_name] = val_best
        metrics["validation"] = val_best_all

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
        "primary_value": metrics.get("validation", {}).get(primary_output, {}).get(primary_metric)
        or metrics.get("test", {}).get(primary_output, {}).get(primary_metric),
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
        # Store all test metrics for the primary output alongside the primary metric
        all_test = result.get("metrics", {}).get("test", {}).get(result.get("dataset", ""), {})
        if not all_test:
            # Fallback: get from any output feature
            test_metrics = result.get("metrics", {}).get("test", {})
            for feat_name, feat_metrics in test_metrics.items():
                if isinstance(feat_metrics, dict):
                    all_test = feat_metrics
                    break

        lb[dataset][model] = {
            "primary_metric": metric,
            metric: value,
            "all_test_metrics": all_test,
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
                pm = entry.get("primary_metric", "")
                val = entry.get(pm)
                if val is not None:
                    row += f"{val:>20.6f}"
                else:
                    row += f"{'—':>20s}"
            else:
                row += f"{'—':>20s}"
        print(row)


def print_comparison(base_dir: str, dataset_name: str):
    """Print comparison table for all models on a single dataset with all metrics."""
    lb = load_leaderboard(base_dir)
    if dataset_name not in lb:
        print(f"No results for dataset '{dataset_name}'.")
        return

    entries = lb[dataset_name]

    # Print primary metric ranking
    primary_metric = None
    for entry in entries.values():
        primary_metric = entry.get("primary_metric")
        if primary_metric:
            break

    print(f"\nDataset: {dataset_name}")
    print(f"Primary metric: {primary_metric}")
    print(f"\n{'Model':<25s} {'Primary':>12s}", end="")

    # Collect all test metric names across all models
    all_metric_names = set()
    for entry in entries.values():
        all_test = entry.get("all_test_metrics", {})
        all_metric_names.update(all_test.keys())
    all_metric_names.discard(primary_metric)
    other_metrics = sorted(all_metric_names)

    for m in other_metrics:
        print(f" {m:>12s}", end="")
    print()
    print("-" * (25 + 13 + 13 * len(other_metrics)))

    for model in sorted(entries, key=lambda m: entries[m].get(primary_metric, 0), reverse=True):
        entry = entries[model]
        pval = entry.get(primary_metric, float("nan"))
        print(f"{model:<25s} {pval:>12.6f}", end="")
        all_test = entry.get("all_test_metrics", {})
        for m in other_metrics:
            val = all_test.get(m)
            if val is not None and isinstance(val, (int, float)):
                print(f" {val:>12.6f}", end="")
            else:
                print(f" {'—':>12s}", end="")
        print()
