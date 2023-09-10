"""
To generate eval metrics, we run inference on our model to generate a test set.
This code generates the images, then converts these images to audio for FAD and IS evaluation
"""

import json
import os
import sys

import torch
from diffusers import StableDiffusionPipeline

# Append parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from data.utils.spectrogram_params import SpectrogramParams
from data.wav_preprocessor import WavPreprocessor


def generate_test_set(model, prompts, output_folder):
    """
    Use a pre-trained model to create tests set through inference
    Args:
        Model: Path to model weight directory (created by our training script)
        Prompts: JSON of test data (containing captions) for inference
        Output_folder: Location to output .png test data
    """
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

    # Prepare output folder
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
        print(f"{youtube_id}.png saved!")


def convert_test_to_audio(image_dir, output_dir):
    """
    Function used to convert .png inference outputs to .wav
    """
# Updated function
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

    # Process entire directory of images to wav
    processor.spec_to_wav_folder(image_dir, output_dir)


def resample_audio(audio_dir):
    """
    Resample audio for evaluation code (which only works with 16kHz and 32kHz audio)
    """
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

    processor.resample_folder(audio_dir, 16000)


checkpoints = [
    "/vol/bitbucket/rp22/sdspeech/evaluation/model-checkpoints/7e4-75epochs-full/7e4-checkpoint-852142-final",
    "/vol/bitbucket/rp22/sdspeech/evaluation/model-checkpoints/5e4-75epochs-full/5e4-checkpoint-852142-final",
]
for chkpt in checkpoints:
    # Extract the desired folder name
    folder_name = chkpt.split("/")[-2].split("-")[0]

    # Construct the image output path
    image_output_path = os.path.join("data/test", folder_name)

    # Generate test set
    generate_test_set(chkpt, "test_captions.json", image_output_path)
    
    # Uncomment the below line if you want to generate test set
    # generate_test_set(chkpt, "test_captions.json", image_output_path)

    # Construct the audio output path
    audio_output_path = os.path.join(image_output_path, "wav")

    # Convert generated images to .wav
    convert_test_to_audio(image_output_path, audio_output_path)

wavs = ["data/test/5e4/wav", "data/test/7e4/wav"]

for wav in wavs:
    resample_audio(wav)
