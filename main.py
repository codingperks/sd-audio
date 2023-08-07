#from config.config import parse_json
#from data.data_pipeline import WavPreprocessor
#from data.utils.spectrogram_params import SpectrogramParams
#from model.sd_ex.lora.train_text_to_image_lora import Sd_model_lora
from data.audiocaps.downloader import download_audiocaps
import os

if __name__ == "__main__":
    # Prepare data
    # add when needed

    """ 
    path = 'data/audiocaps/dataset/train/'
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    number_of_files = len(files)

    print(f"There are {number_of_files} files in the train folder.")

    path = 'data/audiocaps/dataset/test/'
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    number_of_files = len(files)

    print(f"There are {number_of_files} files in the test folder.")

    path = 'data/audiocaps/dataset/val/'
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    number_of_files = len(files)

    print(f"There are {number_of_files} files in the val folder.") """

    """      download_audiocaps(
            '/vol/bitbucket/rp22/sdspeech/data/audiocaps/dataset/train.csv',
            '/vol/bitbucket/rp22/sdspeech/data/audiocaps/dataset/train_download_success.csv',
            '/vol/bitbucket/rp22/sdspeech/data/audiocaps/dataset/train_download_fail.csv',
            '/vol/bitbucket/rp22/sdspeech/data/audiocaps/dataset/train',
            0.1
        ) """

    # Validation dataset
    download_audiocaps(
        '/vol/bitbucket/rp22/sdspeech/data/audiocaps/dataset/val.csv',
        '/vol/bitbucket/rp22/sdspeech/data/audiocaps/dataset/val_download_success.csv',
        '/vol/bitbucket/rp22/sdspeech/data/audiocaps/dataset/val_download_fail.csv',
        '/vol/bitbucket/rp22/sdspeech/data/audiocaps/dataset/val',
        1.0
    )

    # Test dataset
    download_audiocaps(
        '/vol/bitbucket/rp22/sdspeech/data/audiocaps/dataset/test.csv',
        '/vol/bitbucket/rp22/sdspeech/data/audiocaps/dataset/test_download_success.csv',
        '/vol/bitbucket/rp22/sdspeech/data/audiocaps/dataset/test_download_fail.csv',
        '/vol/bitbucket/rp22/sdspeech/data/audiocaps/dataset/test',
        1.0
    )
    
    
    """     # Create processor
    params = SpectrogramParams(
        sample_rate=44100,
        stereo=False,
        step_size_ms=(10 * 1000) / 512,
        min_frequency=20,
        max_frequency=20000,
        num_frequencies=512,
    )
    processor = WavPreprocessor(spectrogram_params=params)

    # Parse JSON parameters
    config = parse_json("config/json/config.json")

    # Create model
    model = Sd_model_lora(preprocessor=processor, **config)

    # train model
    model.train()
    """
