"""
Short script for converting spectrograms in webui-output folders to .wav
"""
import os
import shutil
import sys

# Append the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from PIL import Image

from data.utils.spectrogram_params import SpectrogramParams
from data.wav_preprocessor import WavPreprocessor


def convert_test_to_audio(image_file, output_dir):
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

    # Open the image using PIL
    image = Image.open(image_file)

    # Extract the base name of the image file without its extension
    base_name = os.path.basename(image_file).rsplit(".", 1)[0]

    # Formulate the new path with .wav extension
    output_file = os.path.join(os.path.dirname(image_file), f"{base_name}.wav")

    # Convert the image to audio and export it
    segment = processor.spec_to_wav(image)
    segment.export(output_file, format="wav")

    # Move the resulting .wav file to the desired output directory
    wav_filename = os.path.splitext(os.path.basename(image_file))[0] + ".wav"
    source_path = os.path.join(os.path.dirname(image_file), wav_filename)
    destination_path = os.path.join(output_dir, wav_filename)
    shutil.move(source_path, destination_path)


def process_folder_recursive(current_folder):
    # Create output folder in the current directory
    output_folder_path = os.path.join(current_folder, "wav_outputs")
    if not os.path.exists(output_folder_path):
        os.mkdir(output_folder_path)

    # Convert all png files in the current folder
    png_files = [f for f in os.listdir(current_folder) if f.endswith(".png")]
    for png_file in png_files:
        png_file_path = os.path.join(current_folder, png_file)
        convert_test_to_audio(png_file_path, output_folder_path)

    # Recursively process all subdirectories
    subfolders = [
        d
        for d in os.listdir(current_folder)
        if os.path.isdir(os.path.join(current_folder, d)) and d != "wav_outputs"
    ]
    for subfolder in subfolders:
        process_folder_recursive(os.path.join(current_folder, subfolder))


def process_parent_folder(parent_folder):
    process_folder_recursive(parent_folder)


if __name__ == "__main__":
    parent_folder_path = input("Enter the path of the parent folder: ")
    process_parent_folder(parent_folder_path)
