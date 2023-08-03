from config.config import parse_json
from data.data_pipeline import WavPreprocessor
from data.utils.spectrogram_params import SpectrogramParams
from model.sd_ex.lora.train_text_to_image_lora import Sd_model_lora

if __name__ == "__main__":
    # Prepare data
    # add when needed

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

    # Parse JSON parameters
    config = parse_json("config/json/config.json")

    # Create model
    model = Sd_model_lora(preprocessor=processor, **config)

    # train model
    model.train()
