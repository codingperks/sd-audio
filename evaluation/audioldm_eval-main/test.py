import torch
from audioldm_eval import EvaluationHelper

device = torch.device(f"cuda:{0}")

generation_result_path = "../audiocaps/data/test/7e4/wav"
# generation_result_path = "example/unpaired"
target_audio_path = "../audiocaps/data/test/ground_truth"

evaluator = EvaluationHelper(16000, device)

# Perform evaluation, result will be print out and saved as json
metrics = evaluator.main(
    generation_result_path,
    target_audio_path,
)