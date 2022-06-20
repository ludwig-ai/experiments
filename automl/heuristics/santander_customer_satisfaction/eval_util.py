import numpy as np

def set_model_threshold(model, threshold):
    model.config['output_features'][0]['threshold'] = threshold
    model.model.output_features.module_dict.TARGET__ludwig.threshold = threshold
    model.model.output_features.module_dict.TARGET__ludwig._prediction_module.threshold = threshold

def get_best_threshold(model, dataset, pred_column, metric, threshold_range):
    scores = []
    for threshold in threshold_range:
        set_model_threshold(model, threshold)
        eval_stats, _, _ = model.evaluate(dataset=dataset, split="validation", collect_overall_stats=True)
        metric_score = eval_stats[pred_column]['overall_stats'][metric]
        scores.append(metric_score)
    return threshold_range[np.argmax(scores)]
