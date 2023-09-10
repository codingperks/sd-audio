import json

import torch
from audioldm_eval.audioldm_eval.eval import EvaluationHelper


def evaluate(gen_dir, test_dir, results_name):
    # GPU acceleration is preferred
    device = torch.device(f"cuda:{0}")

    generation_result_path = gen_dir
    target_audio_path = test_dir

    # Initialize a helper instance
    evaluator = EvaluationHelper(44100, device)

    # Perform evaluation, result will be print out and saved as json
    metrics = evaluator.main(
        generation_result_path,
        target_audio_path,
        limit_num=None,  # If you only intend to evaluate X (int) pairs of data, set limit_num=X
    )

    results_name += ".json"

    with open(results_name, "w") as outfile:
        json.dump(metrics, outfile)

evaluate("audiocaps/data/test/5e4", "audiocaps/data/test/ground_truth", "audiocaps/results/5e4")