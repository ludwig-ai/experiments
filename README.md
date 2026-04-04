# Ludwig Experiments

Standardized experiment harness for benchmarking Ludwig models across datasets.

## Structure

```
datasets/           # One dir per dataset with dataset.yaml
models/             # One dir per model with model.yaml (combiner + trainer config)
harness/            # Python package: runner, registry, results management
results/            # Collected results (runs + leaderboard)
```

## Quick Start

```bash
# Run one model on one dataset
python -m harness.runner --dataset adult_census_income --model ft_transformer

# Run all models on one dataset
python -m harness.runner --dataset adult_census_income

# Run one model on all datasets
python -m harness.runner --model ft_transformer

# Run everything
python -m harness.runner

# Multi-seed for statistical significance
python -m harness.runner --dataset adult_census_income --model ft_transformer --seeds 42 43 44 45 46

# View leaderboard
python -m harness.runner --leaderboard

# Compare models on a dataset
python -m harness.runner --compare adult_census_income
```

## Adding a Dataset

Create `datasets/{name}/dataset.yaml`:

```yaml
name: my_dataset
loader: ludwig              # ludwig, sklearn, or csv
loader_args:
  module: my_dataset        # ludwig.datasets module name
task: binary_classification
primary_metric: roc_auc
primary_output: target_col
minimize: false
input_features:
  - name: feature1
    type: number
output_features:
  - name: target_col
    type: binary
```

## Adding a Model

Create `models/{name}/model.yaml`:

```yaml
name: my_model
description: "My custom model configuration"
combiner:
  type: ft_transformer
  hidden_size: 192
trainer:
  learning_rate: 0.0001
  batch_size: 256
encoder_overrides:       # Optional: apply encoder to all features of a type
  number:
    type: ple
    num_bins: 64
```

## Available Datasets

| Dataset | Task | Metric | Features |
|---------|------|--------|----------|
| adult_census_income | Binary classification | ROC AUC | 14 mixed |
| california_housing | Regression | RMSE | 8 numerical |
| higgs | Binary classification | ROC AUC | 28 numerical |

## Available Models

| Model | Source Paper | Combiner |
|-------|------------|----------|
| concat_baseline | — | ConcatCombiner |
| transformer | — | TransformerCombiner |
| ft_transformer | Gorishniy et al., NeurIPS 2021 | FTTransformerCombiner |
| cross_attention | Cross-attention literature | CrossAttentionCombiner |
| perceiver | Jaegle et al., ICML 2022 | PerceiverCombiner |
| gated_fusion | Alayrac et al., NeurIPS 2022 | GatedFusionCombiner |
| ple_concat | Gorishniy et al., NeurIPS 2022 | PLE encoder + Concat |
| periodic_concat | Gorishniy et al., NeurIPS 2022 | Periodic encoder + Concat |
