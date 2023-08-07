import shutil
from datetime import date

from config.config import parse_json
from data.utils.spectrogram_params import SpectrogramParams

# from data.ac_data_pipeline import ACPipeline
from data.wav_preprocessor import WavPreprocessor
from model.sd_ex.lora.train_text_to_image_lora import Sd_model_lora

if __name__ == "__main__":
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

    # Prepare data - uncomment if needed
    # pipeline = ACPipeline("./data/audiocaps/dataset", processor)
    # pipeline.create_dataset(44100)

    # Parse JSON parameters
    config = parse_json("config/json/config.json")

    # CLI JSON Parsing
    print("Select parameters")
    lr = input("Learning rate: ")
    steps = input("Training steps: ")
    warmup = input("Warmup steps: ")

    # Update config parameters
    today = date.today()
    date = today.strftime("%m-%d")
    config["output_dir"] += date + "/" + lr + "_" + steps + "steps_" + warmup + "warmup"

    if lr != "":
        config["learning_rate"] = float(lr)
    if steps != "":
        config["max_train_steps"] = int(steps)
    if warmup != "":
        config["lr_warmup_steps"] = int(warmup)

    print(f"Learning rate:{config['learning_rate']}")
    print(f"Training steps: {config['max_train_steps']}")
    print(f"Warmup steps: {config['lr_warmup_steps']}")

    # Create model
    model = Sd_model_lora(preprocessor=processor, **config)

    # Save config in output folder
    shutil.copy("config/json/config.json", config["output_dir"])

    # train model
    model.train()
