import csv
import os


class ACPipeline:
    """
    Pipeline for processing audiocaps data - applying preprocessing steps + generating metadata.csv
    """

    def __init__(self, dataset_path, preprocessor):
        self._dataset_path = dataset_path  # Where data is saved and constructed
        self._preprocessor = preprocessor

    # Apply pre-processing steps to data
    def preprocess(self, target_sr):
        for folder in os.listdir(self._dataset_path):
            if folder in ["test", "train", "val"]:
                folder_path = os.path.join(
                    self._dataset_path, folder
                )  # Full path to the folder
                self._preprocessor.resample_folder(folder_path, target_sr)
                self._preprocessor.adjust_length_folder(folder_path)
                self._preprocessor.min_max_normalise_folder(folder_path)
                self._preprocessor.wav_to_spec_folder(folder_path)

    def generate_metadata(self, target_split, success_file):
        # for selected folder
        # create metadata of [image, caption, audiofile(path)]
        # match filename with caption
        data_folder = os.path.join(self._dataset_path, target_split)
        metadata = data_folder + "/metadata.csv"

        with open(success_file, "r") as infile, open(
            metadata, "w", newline=""
        ) as outfile:
            reader = csv.DictReader(infile)
            fieldnames = ["image", "caption", "audio"]
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)

            writer.writeheader()

            for row in reader:
                youtube_id = row["youtube_id"]
                wav_file = os.path.join(data_folder, f"{youtube_id}.wav")

                if os.path.exists(wav_file):
                    writer.writerow(
                        {
                            "file_name": f"{youtube_id}.png",
                            "text": "a spectrogram of " + row["caption"],
                            "audiofile": wav_file,
                        }
                    )

    def test_convert_to_wav(self, output_folder):
        unsplit_path = os.path.join(self._dataset_path, "dataset/unsplit")

        self._preprocessor.spec_to_wav_folder(unsplit_path, output_folder)
        print("Conversion spec to wav complete")

    def test_convert_to_wav_np(self, output_folder):
        unsplit_path = os.path.join(self._dataset_path, "dataset/unsplit")

        self._preprocessor.spec_to_wav_np_folder(unsplit_path, output_folder)
        print("Conversion spec to wav complete")

    def create_dataset(self, target_sr):
        # 1. Apply preprocessing (resampling and min_max_norm) for each class and create spectrograms
        self.preprocess(target_sr)

        # 2. Generate metadata for each class
        self.generate_metadata(
            "train", "data/audiocaps/dataset_full/train_download_success.csv"
        )
        self.generate_metadata(
            "test", "data/audiocaps/dataset_full/test_download_success.csv"
        )
        self.generate_metadata(
            "val", "data/audiocaps/dataset_full/val_download_success.csv"
        )

        # 3. Test conversion back
        """         os.makedirs(self._dataset_path + "/dataset/spec_to_wav_test", exist_ok=True)
        os.makedirs(self._dataset_path + "/dataset/spec_to_wav_np_test", exist_ok=True)

        self.test_convert_to_wav(self._dataset_path + "/dataset/spec_to_wav_test")
        self.test_convert_to_wav_np(self._dataset_path + "/dataset/spec_to_wav_np_test") """
