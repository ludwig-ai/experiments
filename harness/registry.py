"""Registry for dataset and model configs.

Loads YAML configs from the datasets/ and models/ directories and merges them into
full Ludwig configs ready for training.
"""

import copy
import os
from pathlib import Path

import yaml


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def find_configs(base_dir: str, subdir: str) -> dict[str, dict]:
    """Discover all configs in a directory (one YAML per subdirectory)."""
    configs = {}
    config_dir = os.path.join(base_dir, subdir)
    if not os.path.isdir(config_dir):
        return configs
    for name in sorted(os.listdir(config_dir)):
        yaml_path = os.path.join(config_dir, name, f"{subdir.rstrip('s')}.yaml")  # dataset.yaml or model.yaml
        if os.path.isfile(yaml_path):
            configs[name] = load_yaml(yaml_path)
    return configs


class Registry:
    """Registry of datasets and models with config merging."""

    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.datasets = find_configs(base_dir, "datasets")
        self.models = find_configs(base_dir, "models")

    def list_datasets(self) -> list[str]:
        return list(self.datasets.keys())

    def list_models(self) -> list[str]:
        return list(self.models.keys())

    def get_dataset(self, name: str) -> dict:
        if name not in self.datasets:
            raise ValueError(f"Unknown dataset '{name}'. Available: {self.list_datasets()}")
        return copy.deepcopy(self.datasets[name])

    def get_model(self, name: str) -> dict:
        if name not in self.models:
            raise ValueError(f"Unknown model '{name}'. Available: {self.list_models()}")
        return copy.deepcopy(self.models[name])

    def build_ludwig_config(self, dataset_name: str, model_name: str) -> dict:
        """Merge dataset and model configs into a full Ludwig config.

        Dataset provides: input_features, output_features, preprocessing
        Model provides: combiner, trainer, encoder_overrides
        """
        dataset = self.get_dataset(dataset_name)
        model = self.get_model(model_name)

        config = {
            "input_features": dataset["input_features"],
            "output_features": dataset["output_features"],
        }

        # Dataset preprocessing
        if "preprocessing" in dataset:
            config["preprocessing"] = dataset["preprocessing"]

        # Model combiner
        if "combiner" in model:
            config["combiner"] = model["combiner"]

        # Model trainer
        if "trainer" in model:
            config["trainer"] = model["trainer"]

        # Encoder overrides: apply model-specified encoders to features by type
        encoder_overrides = model.get("encoder_overrides", {})
        for feature in config["input_features"]:
            ftype = feature["type"]
            if ftype in encoder_overrides and "encoder" not in feature:
                feature["encoder"] = copy.deepcopy(encoder_overrides[ftype])

        # Backend: default to local
        config["backend"] = {"type": "local"}

        return config
