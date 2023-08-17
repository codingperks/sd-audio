import argparse
import shutil
from datetime import date

from config.config import parse_json

# from data.ac_data_pipeline import ACPipeline
from data.utils.spectrogram_params import SpectrogramParams
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
    # pipeline = ACPipeline.create_dataset()

    # Parse JSON parameters
    config = parse_json("config/json/config.json")

    # Argparse setup
    parser = argparse.ArgumentParser(
        description="Train the model with custom parameters."
    )
    parser.add_argument(
        "--lr", type=float, help="Learning rate", default=config["learning_rate"]
    )
    parser.add_argument(
        "--steps", type=int, help="Training steps", default=config["max_train_steps"]
    )
    parser.add_argument(
        "--warmup", type=int, help="Warmup steps", default=config["lr_warmup_steps"]
    )
    parser.add_argument(
        "--adamw", type=int, help="Adam weight decay", default=config["adam_weight_decay"]
    )
    parser.add_argument(
        "--adam2", type=int, help="Adam b2", default=config["adam_beta2"]
    )

    args = parser.parse_args()

    # Update config parameters
    today = date.today()
    date_str = today.strftime("%m-%d")
    config[
        "output_dir"
    ] += f"{date_str}/{args.lr}_{args.steps}steps_{args.warmup}warmup"
    config_dir = config["output_dir"].replace(".", "")

    config["learning_rate"] = args.lr
    config["max_train_steps"] = args.steps
    config["lr_warmup_steps"] = args.warmup
    
    if args.adamw:
        config["adam_weight_decay"] = args.lr
    if args.adam2:
        config["adam_beta2"] = args.lr

    print(f"Learning rate:{config['learning_rate']}")
    print(f"Training steps: {config['max_train_steps']}")
    print(f"Warmup steps: {config['lr_warmup_steps']}")
    print(f"Warmup steps: {config['lr_warmup_steps']}")
    print(f"Adam weight decay: {config['adam_weight_decay']}")
    print(f"Adam beta 2: {config['adam_beta2']}")

    # Create model
    model = Sd_model_lora(preprocessor=processor, **config)

    # train model
    model.train()

    # Save config in output folder
    shutil.copy("config/json/config.json", config["output_dir"])
