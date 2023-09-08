import torch
from audioldm_eval.audioldm_eval-main import EvaluationHelper

def evaluate(gen_dir, test_dir):
    # GPU acceleration is preferred
    device = torch.device(f"cuda:{0}")

    generation_result_path = gen_data
    target_audio_path = test_dir

    # Initialize a helper instance
    evaluator = EvaluationHelper(44100, device)

    # Perform evaluation, result will be print out and saved as json
    metrics = evaluator.main(
        generation_result_path,
        target_audio_path,
        limit_num=None # If you only intend to evaluate X (int) pairs of data, set limit_num=X
    )
