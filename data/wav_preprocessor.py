import os

import librosa
import numpy as np
import pydub
import soundfile as sf

from data.utils.spectrogram_image_converter import SpectrogramImageConverter


class WavPreprocessor:
    """
    Preprocessor object for processing wav files: resampling, normalising, padding and converting to spec
    """

    def __init__(self, spectrogram_params):
        self._params = spectrogram_params
        self._converter = SpectrogramImageConverter(
            params=spectrogram_params, device="cuda"
        )

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
