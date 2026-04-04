"""Experiment runner: run models on datasets in any combination.

Usage:
    python -m harness.runner --dataset adult_census_income --model ft_transformer
    python -m harness.runner --dataset adult_census_income --model concat_baseline ft_transformer
    python -m harness.runner --all
    python -m harness.runner --leaderboard
    python -m harness.runner --compare adult_census_income
"""

import argparse
import concurrent.futures
import logging
import os
import sys
import tempfile
import time

from harness.registry import Registry
from harness.results import (
    build_result,
    print_comparison,
    print_leaderboard,
    save_result,
    update_leaderboard,
)

logger = logging.getLogger(__name__)


def load_dataset(dataset_config: dict):
    """Load a dataset based on its config."""
    import pandas as pd

    loader = dataset_config.get("loader", "ludwig")
    loader_args = dataset_config.get("loader_args", {})

    if loader == "ludwig":
        module_name = loader_args.get("module", dataset_config["name"])
        from ludwig.datasets import get_dataset

        ds = get_dataset(module_name)
        df = ds.load()
        return df
    elif loader == "sklearn":
        func_name = loader_args["function"]
        import sklearn.datasets

        fetch_fn = getattr(sklearn.datasets, func_name)
        data = fetch_fn()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        target_col = loader_args.get("target_column", "target")
        df[target_col] = data.target
        # Add split column
        import numpy as np

        n = len(df)
        splits = np.zeros(n, dtype=int)
        splits[int(n * 0.7) : int(n * 0.85)] = 1
        splits[int(n * 0.85) :] = 2
        df["split"] = splits
        return df
    elif loader == "csv":
        return pd.read_csv(loader_args["path"])
    else:
        raise ValueError(f"Unknown loader: {loader}")


class ExperimentRunner:
    """Run Ludwig experiments across datasets and models."""

    def __init__(self, base_dir: str = None):
        if base_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.base_dir = base_dir
        self.registry = Registry(base_dir)

    @staticmethod
    def _get_num_gpus() -> int:
        """Detect number of available GPUs."""
        try:
            import torch

            if torch.cuda.is_available():
                return torch.cuda.device_count()
        except ImportError:
            pass
        return 0

    def run(
        self,
        datasets: str | list[str] = None,
        models: str | list[str] = None,
        seeds: list[int] = None,
        epochs_override: int = None,
        gpu: bool = True,
        max_parallel: int = None,
    ) -> list[dict]:
        """Run experiments for given dataset/model combinations.

        When multiple GPUs are available, runs up to N experiments in parallel (one per GPU).
        Each experiment is pinned to a specific GPU via CUDA_VISIBLE_DEVICES.

        Args:
            datasets: Dataset name(s). None = all registered datasets.
            models: Model name(s). None = all registered models.
            seeds: Random seeds to run. Default [42].
            epochs_override: Override epochs in trainer config.
            gpu: Whether to use GPU if available.
            max_parallel: Max parallel experiments. Default = number of GPUs (or 1 if CPU).

        Returns:
            List of result dicts.
        """
        if datasets is None:
            datasets = self.registry.list_datasets()
        elif isinstance(datasets, str):
            datasets = [datasets]

        if models is None:
            models = self.registry.list_models()
        elif isinstance(models, str):
            models = [models]

        if seeds is None:
            seeds = [42]

        # Build job list
        jobs = []
        for dataset_name in datasets:
            for model_name in models:
                for seed in seeds:
                    jobs.append((dataset_name, model_name, seed))

        total = len(jobs)
        num_gpus = self._get_num_gpus() if gpu else 0
        if max_parallel is None:
            max_parallel = max(num_gpus, 1)

        print(f"Running {total} experiments ({max_parallel} parallel, {num_gpus} GPUs detected)")

        if max_parallel <= 1:
            # Sequential execution
            return self._run_sequential(jobs, epochs_override, gpu)
        else:
            # Parallel execution with GPU pinning
            return self._run_parallel(jobs, epochs_override, num_gpus, max_parallel)

    def _run_sequential(self, jobs, epochs_override, gpu) -> list[dict]:
        """Run jobs one at a time."""
        results = []
        for i, (dataset_name, model_name, seed) in enumerate(jobs, 1):
            print(f"\n[{i}/{len(jobs)}] {dataset_name} / {model_name} (seed={seed})")
            try:
                result = self._run_one(dataset_name, model_name, seed, epochs_override, gpu)
                results.append(result)
                self._save_and_report(result, dataset_name)
            except Exception as e:
                print(f"  FAILED: {e}")
                logger.exception(f"Experiment failed: {dataset_name}/{model_name}")
        return results

    def _run_parallel(self, jobs, epochs_override, num_gpus, max_parallel) -> list[dict]:
        """Run jobs in parallel, pinning each to a specific GPU."""
        results = []
        gpu_semaphores = list(range(num_gpus)) if num_gpus > 0 else [None]

        def run_with_gpu(args):
            idx, (dataset_name, model_name, seed) = args
            gpu_id = gpu_semaphores[idx % len(gpu_semaphores)]
            if gpu_id is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            print(f"  [{idx + 1}/{len(jobs)}] {dataset_name}/{model_name} seed={seed} gpu={gpu_id}")
            return self._run_one(dataset_name, model_name, seed, epochs_override, gpu=gpu_id is not None)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as executor:
            futures = {
                executor.submit(run_with_gpu, (i, job)): job for i, job in enumerate(jobs)
            }
            for future in concurrent.futures.as_completed(futures):
                dataset_name, model_name, seed = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    self._save_and_report(result, dataset_name)
                except Exception as e:
                    print(f"  FAILED {dataset_name}/{model_name}: {e}")
                    logger.exception(f"Experiment failed: {dataset_name}/{model_name}")

        return results

    def _save_and_report(self, result, dataset_name):
        """Save result and update leaderboard."""
        path = save_result(result, self.base_dir)
        ds_config = self.registry.get_dataset(dataset_name)
        is_minimize = ds_config.get("minimize", False)
        update_leaderboard(result, self.base_dir, is_minimize)
        val = result.get("primary_value")
        metric = result.get("primary_metric", "?")
        elapsed = result.get("training_time_seconds", 0)
        print(f"  {metric}={val:.6f} ({elapsed:.1f}s) -> {path}")

    def _run_one(
        self,
        dataset_name: str,
        model_name: str,
        seed: int,
        epochs_override: int | None,
        gpu: bool,
    ) -> dict:
        """Run a single experiment."""
        from ludwig.api import LudwigModel

        # Build config
        config = self.registry.build_ludwig_config(dataset_name, model_name)

        if epochs_override:
            config.setdefault("trainer", {})["epochs"] = epochs_override

        if not gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

        # Load dataset
        ds_config = self.registry.get_dataset(dataset_name)
        df = load_dataset(ds_config)

        # Train
        start = time.time()
        with tempfile.TemporaryDirectory() as tmpdir:
            model = LudwigModel(config=config, logging_level=logging.WARNING)
            train_stats, _, _ = model.train(dataset=df, output_directory=tmpdir, random_seed=seed)

            # Evaluate on test set
            eval_stats, _, _ = model.evaluate(dataset=df, output_directory=tmpdir)

        elapsed = time.time() - start

        # Build result
        primary_metric = ds_config.get("primary_metric", "loss")
        primary_output = ds_config.get("primary_output", ds_config["output_features"][0]["name"])

        return build_result(
            dataset_name=dataset_name,
            model_name=model_name,
            config=config,
            train_stats=train_stats,
            eval_stats=eval_stats,
            training_time=elapsed,
            seed=seed,
            primary_metric=primary_metric,
            primary_output=primary_output,
        )

    def leaderboard(self):
        print_leaderboard(self.base_dir)

    def compare(self, dataset_name: str):
        print_comparison(self.base_dir, dataset_name)


def main():
    parser = argparse.ArgumentParser(description="Ludwig Experiment Harness")
    parser.add_argument("--dataset", nargs="*", help="Dataset name(s). Omit for all.")
    parser.add_argument("--model", nargs="*", help="Model name(s). Omit for all.")
    parser.add_argument("--seeds", nargs="*", type=int, default=[42], help="Random seeds.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs.")
    parser.add_argument("--no-gpu", action="store_true", help="Force CPU.")
    parser.add_argument("--max-parallel", type=int, default=None, help="Max parallel experiments (default: num GPUs).")
    parser.add_argument("--leaderboard", action="store_true", help="Print leaderboard.")
    parser.add_argument("--compare", type=str, help="Compare models on a dataset.")
    parser.add_argument("--base-dir", type=str, default=None, help="Experiments repo root.")
    args = parser.parse_args()

    if args.base_dir:
        base_dir = args.base_dir
    else:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    runner = ExperimentRunner(base_dir=base_dir)

    if args.leaderboard:
        runner.leaderboard()
        return

    if args.compare:
        runner.compare(args.compare)
        return

    runner.run(
        datasets=args.dataset,
        models=args.model,
        seeds=args.seeds,
        epochs_override=args.epochs,
        gpu=not args.no_gpu,
        max_parallel=args.max_parallel,
    )

    print("\n" + "=" * 60)
    print("LEADERBOARD")
    print("=" * 60)
    runner.leaderboard()


if __name__ == "__main__":
    main()
