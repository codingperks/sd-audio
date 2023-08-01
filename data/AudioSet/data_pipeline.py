import os
import librosa
import soundfile as sf
import numpy as np
import sys

sys.path.append("..")
from utils.spectrogram_image_converter import SpectrogramImageConverter
from utils.spectrogram_params import SpectrogramParams
import pydub
import typing as T
from PIL import Image
import shutil
import csv
import random
import math


class WavPreprocessor:
    def __init__(self, spectrogram_params):
        self._params = spectrogram_params
        self._converter = SpectrogramImageConverter(
            params=spectrogram_params, device="cuda"
        )

        return

    def resample(self, audio, target_sr):
        y, sr = librosa.load(audio, sr=None)

        y_resampled = librosa.resample(y=y, orig_sr=sr, target_sr=target_sr)

        return y_resampled, target_sr
    
    def resample_folder(self, input_path, target_sr):
        for filename in os.listdir(input_path):
            if filename.endswith(".wav"):
                path = os.path.join(input_path, filename)
                y_resampled, sr = self.resample(path, target_sr)

                save_path = os.path.join(
                    input_path, filename
                )  # Overwrite the original file
                sf.write(save_path, y_resampled, sr)

    def min_max_normalise(self, audio):
        y, sr = librosa.load(audio, sr=None)

        normalised_y = y / np.max(np.abs(y))

        return normalised_y, sr

    def adjust_length(self, audio_clip, sample_rate):
        desired_length = 10 * sample_rate  # 10 seconds * sample_rate

        # If the audio clip is longer than 10 seconds, shorten it
        if len(audio_clip) > desired_length:
            return audio_clip[:desired_length]

        # If the audio clip is shorter than 10 seconds, pad it with zeros
        elif len(audio_clip) < desired_length:
            return np.pad(audio_clip, (0, desired_length - len(audio_clip)))

        # If the audio clip is exactly 10 seconds, return it as is
        else:
            return audio_clip

    def adjust_length_folder(self, input_path):
        for filename in os.listdir(input_path):
            if filename.endswith(".wav"):
                path = os.path.join(input_path, filename)
                audio_clip, sr = librosa.load(path, sr=None)
                adjusted_audio = self.adjust_length(audio_clip, sr)

                save_path = os.path.join(
                    input_path, filename
                )  # Overwrite the original file

                sf.write(save_path, adjusted_audio, sr)

    def min_max_normalise_folder(self, input_path):
        for filename in os.listdir(input_path):
            if filename.endswith(".wav"):
                path = os.path.join(input_path, filename)
                norm_wav, sr = self.min_max_normalise(path)

                save_path = os.path.join(
                    input_path, filename
                )  # Overwrite the original file
                sf.write(save_path, norm_wav, sr)

    def wav_to_spec(self, wav_path):
        # Convert wav to audiosegment
        segment = pydub.AudioSegment.from_wav(wav_path)

        # Convert to mono
        segment = segment.set_channels(1)

        # Generate the spectrogram
        image = self._converter.spectrogram_image_from_audio(segment)


        # Crop width to 512
        image = image.crop((0, 0, 512, 512))

        return image

    def wav_to_spec_folder(self, input_path):
        for wav_file in os.listdir(input_path):
            if wav_file.endswith(".wav"):  # Ensure we're only working on wav files
                wav_path = os.path.join(input_path, wav_file)
                image = self.wav_to_spec(wav_path)  # Output to the same folder
                image_out = os.path.join(
                    input_path, os.path.basename(wav_path)[:-4] + ".png"
                )
                image.save(image_out, exif=image.getexif(), format="PNG")
                print(f"Saved {image_out}")

    def spec_to_wav(self, spec_path):
        # Convert path to image
        # if spec_path is not isinstance(spec_path, Image.Image):
        #    spec_path = Image.open(spec_path)

        # Convert segment to image
        segment = self._converter.audio_from_spectrogram_image(image=spec_path)

        return segment

    def spec_to_wav_np(self, spec_path):
        # Convert path to image
        # if spec_path is not isinstance(spec_path, Image.Image):
        #    spec_path = Image.open(spec_path)

        # Convert segment to image
        segment = self._converter.audio_from_spectrogram_image(image=spec_path)
        segment = segment.get_array_of_samples()
        segment = np.array(segment)

        return segment

    def spec_to_wav_folder(self, input_path, output_path):
        for spec_file in os.listdir(input_path):
            if spec_file.endswith(".png"):  # Ensure we're only working on png files
                spec_path = os.path.join(input_path, spec_file)
                segment = self.spec_to_wav(spec_path)
                audio_out = os.path.join(
                    output_path, os.path.basename(spec_path)[:-4] + ".wav"
                )
                segment.export(audio_out, format="wav")
                print(f"Saved {audio_out}")


# Select chosen classes and copy across into data folder
class DatasetPipeline:
    def __init__(self, dataset_path, class_path, preprocessor, *classes):
        self._dataset_path = dataset_path  # Where dataset will be constructed
        self._class_path = class_path  # Where raw data is saved
        self._classes = [c for c in classes]
        self._preprocessor = preprocessor

    # Clears folders for regeneration
    def reset(self):
        return

    # Creates corresponding folders for chosen classes
    def folder_setup(self):
        for folder in self._classes:
            dest_folder = os.path.join(self._dataset_path, folder)
            if not os.path.exists(dest_folder):
                print(f"Creating directory: {dest_folder}")
                os.makedirs(dest_folder)
            else:
                print(f"Directory already exists: {dest_folder}")

    # Copies files excluding those tagged with REMOVE
    def copy_files(self):
        for folder in os.listdir(self._class_path):
            if folder in self._classes and os.path.isdir(
                os.path.join(self._class_path, folder)
            ):
                for file in os.listdir(os.path.join(self._class_path, folder)):
                    if "REMOVE" in file:
                        continue

                    # Add the folder name (class name) as prefix to each file
                    new_file_name = f"{folder}_{file}"

                    src_file = os.path.join(self._class_path, folder, file)
                    dest_file = os.path.join(self._dataset_path, folder, new_file_name)

                    shutil.copy(src_file, dest_file)

    # Generates a metadata.csv with headings [image, prompt, audiofile]
    def generate_metadata(self):
        # Define the output path for the CSV file
        csv_path = os.path.join(
            self._dataset_path, "dataset", "unsplit", "metadata.csv"
        )

        # Create the CSV file and write the header
        with open(csv_path, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["file_name", "text", "audiofile"])

            # Get the list of PNG files in the source folder
            png_files = [
                f
                for f in os.listdir(
                    os.path.join(self._dataset_path, "dataset", "unsplit")
                )
                if f.endswith(".png")
            ]

            # Iterate over each PNG file
            for png_file in png_files:
                # Extract the filename without the extension
                file_name = os.path.splitext(png_file)[0]

                # Construct the corresponding WAV file path
                audio_file = file_name + ".wav"

                # Construct the prompt
                prompt = f"A spectrogram of {(file_name.split('_')[0])}"

                # Write the row to the CSV file
                writer.writerow([png_file, prompt, audio_file])

    # Combines all classes into a single dataset folder, combining the metadata
    def combine_data(self):
        combined_folder = os.path.join(self._dataset_path, "dataset", "unsplit")
        os.makedirs(combined_folder, exist_ok=True)

        combined_metadata_path = os.path.join(combined_folder, "metadata.csv")
        with open(combined_metadata_path, "w", newline="") as combined_metadata:
            writer = csv.writer(combined_metadata)
            writer.writerow(
                ["file_name", "text", "audiofile"]
            )  # Write header to the combined metadata

            for folder in self._classes:
                class_folder = os.path.join(self._dataset_path, folder)

                # Copy files
                for file in os.listdir(class_folder):
                    if file.endswith(".wav") or file.endswith(".png"):
                        src_file = os.path.join(class_folder, file)
                        dest_file = os.path.join(combined_folder, file)
                        shutil.copy(src_file, dest_file)

                # Append metadata
                metadata_path = os.path.join(class_folder, "metadata.csv")
                with open(metadata_path, "r") as metadata:
                    reader = csv.reader(metadata)
                    next(reader)  # Skip header
                    for row in reader:
                        writer.writerow(row)  # Write each row to the combined metadata

    # Splits data into train, test and var folders
    def split_data(self, train_ratio=0.8, val_ratio=0.1):
        # Base directory of dataset
        base_path = os.path.join(self._dataset_path, "dataset", "unsplit")

        # Create the new directories if they don't exist
        train_folder = os.path.join(self._dataset_path, "dataset", "train")
        val_folder = os.path.join(self._dataset_path, "dataset", "val")
        test_folder = os.path.join(self._dataset_path, "dataset", "test")

        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(val_folder, exist_ok=True)
        os.makedirs(test_folder, exist_ok=True)

        # Collect all unique file prefixes in the unsplit directory
        file_prefixes = {
            filename.split(".")[0]
            for filename in os.listdir(base_path)
            if filename.endswith((".wav", ".png"))
        }
        file_prefixes = list(file_prefixes)

        # Shuffle the list for randomness
        random.shuffle(file_prefixes)

        # Calculate the indices to split at
        total_files = len(file_prefixes)
        train_split = math.floor(total_files * train_ratio)
        val_split = train_split + math.floor(total_files * val_ratio)

        # Split the list
        train_files = file_prefixes[:train_split]
        val_files = file_prefixes[train_split:val_split]
        test_files = file_prefixes[val_split:]

        # Define a helper function to move files
        def move_files(files, destination_folder):
            for file_prefix in files:
                for extension in [".wav", ".png"]:
                    file_name = file_prefix + extension
                    src_path = os.path.join(base_path, file_name)
                    dest_path = os.path.join(destination_folder, file_name)
                    shutil.copy(src_path, dest_path)

        # Move the files
        move_files(train_files, train_folder)
        move_files(val_files, val_folder)
        move_files(test_files, test_folder)

    # Generates a metadata.csv with headings [image, prompt, audiofile]
    def generate_split_metadata(self, split_folder):
        # Define the output path for the CSV file
        csv_path = os.path.join(
            self._dataset_path, "dataset", split_folder, "metadata.csv"
        )

        # Open the CSV file and write the header
        with open(csv_path, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["file_name", "text", "audiofile"])

            # Iterate over each PNG file in the split directory
            for png_file in os.listdir(
                os.path.join(self._dataset_path, "dataset", split_folder)
            ):
                if png_file.endswith(".png"):
                    # Extract the filename without the extension
                    file_name = os.path.splitext(png_file)[0]

                    # Construct the corresponding WAV file name
                    audio_file = file_name + ".wav"

                    # Get the absolute path of the audio file
                    abs_audio_file = os.path.abspath(
                        os.path.join(
                            self._dataset_path, "dataset", split_folder, audio_file
                        )
                    )

                    # Construct the prompt
                    prompt = f"A spectrogram of {(file_name.split('_')[0])}"

                    # Write the row to the CSV file
                    writer.writerow([png_file, prompt, abs_audio_file])

    # Apply pre-processing steps to data
    def preprocess(self, target_sr):
        for folder in os.listdir(self._dataset_path):
            if folder not in self._classes:
                continue
            folder_path = os.path.join(
                self._dataset_path, folder
            )  # Full path to the folder
            self._preprocessor.resample_folder(folder_path, target_sr)
            self._preprocessor.adjust_length_folder(folder_path)
            self._preprocessor.min_max_normalise_folder(folder_path)
            self._preprocessor.wav_to_spec_folder(folder_path)

    def test_convert_to_wav(self, output_folder):
        unsplit_path = os.path.join(self._dataset_path, "dataset/unsplit")

        self._preprocessor.spec_to_wav_folder(unsplit_path, output_folder)
        print("Conversion spec to wav complete")

    def create_dataset(self, target_sr, train_split, var_split):
        assert (
            train_split + var_split <= 1
        ), "Train and validation split must be less than 1"

        # 1. Create folders for each class
        self.folder_setup()

        # 2. Copy files for each class
        self.copy_files()

        # 3. Apply preprocessing (resampling and min_max_norm) for each class and create spectrograms
        self.preprocess(target_sr)

        # 4. Generate metadata for each class
        self.generate_metadata()

        # 5. Combine all classes into a single dataset folder
        self.combine_data()

        # 6. Split data into train, test and validation sets
        self.split_data(train_ratio=train_split, val_ratio=var_split)

        # 7. Create new metadata
        self.generate_split_metadata("train")
        self.generate_split_metadata("val")
        self.generate_split_metadata("test")

        # 8. Test conversion back
        self.test_convert_to_wav(self._dataset_path + "/dataset/spec_to_wav_test")
