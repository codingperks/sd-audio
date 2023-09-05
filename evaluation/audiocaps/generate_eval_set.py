import json
import os

import torch
from diffusers import StableDiffusionPipeline

from data.utils.spectrogram_params import SpectrogramParams
from data.wav_preprocessor import WavPreprocessor


def generate_test_set(model, prompts, output_folder):
    device = "cuda"

    # load model and weights
    model_path = model

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )

    pipe.unet.load_attn_procs(model_path, weight_name="pytorch_model.bin")

    # prepare output folder
    os.makedirs(output_folder, exist_ok=True)

    pipe.to(device)

    # Load the JSON
    with open(prompts, "r") as f:
        prompts = json.load(f)

    for prompt in prompts:
        caption = prompt.get("caption")
        youtube_id = prompt.get("youtube_id")

        # Generate image
        image = pipe(caption).images[0]

        # Save the image
        image.save(os.path.join(output_folder, f"{youtube_id}.png"))


def convert_test_to_audio(image_dir, output_dir):
    # Create processor
    params = SpectrogramParams(
        sample_rate=44100,
        stereo=False,
        step_size_ms=(10 * 1000) / 512,
        min_frequency=20,
        max_frequency=20000,
        num_frequencies=512,
    )
    processor = WavPreprocessor(spectrogram_params=params)

    processor.spec_to_wav_folder(image_dir, output_dir)


generate_test_set(
    "/vol/bitbucket/rp22/sdspeech/model/sd_ex/lora/out/09-04/0.0007_832000steps_28000warmup/checkpoint-250000",
    "test_captions.json",
    "data/test/image",
)
