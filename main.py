"""
This main file creates a WavPreprocessor object for converting between image and audio during data preparation and training.
This file then parses command line arguments to configure our training runs.
This file is ran using train.sh in order to appropriately set environment variables (to avoid overfilling temporary folders).
"""

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
    # pipeline = ACPipeline("./data/audiocaps/dataset_full", processor)
    # pipeline.create_dataset(44100)

    # Parse JSON parameters from config file
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
        "--adamw",
        type=float,
        help="Adam weight decay",
        default=config["adam_weight_decay"],
    )
    parser.add_argument(
        "--adam2", type=float, help="Adam b2", default=config["adam_beta2"]
    )
    parser.add_argument(
        "--batch",
        type=int,
        help="Train and val batch sizes",
        default=config["train_batch_size"],
    )
    parser.add_argument(
        "--full", type=bool, help="True if using full dataset", default=False
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
        config["adam_weight_decay"] = args.adamw
    if args.adam2:
        config["adam_beta2"] = args.adam2
    if args.batch:
        config["train_batch_size"] = args.batch
        config["val_batch_size"] = args.batch
    if args.full:
        config["train_data_dir"] = "./data/audiocaps/dataset_full/train"
        config["val_data_dir"] = "./data/audiocaps/dataset_full/val"
        print("Using full AC dataset")

    print(f"Learning rate:{config['learning_rate']}")
    print(f"Training steps: {config['max_train_steps']}")
    print(f"Warmup steps: {config['lr_warmup_steps']}")
    print(f"Adam weight decay: {config['adam_weight_decay']}")
    print(f"Adam beta 2: {config['adam_beta2']}")
    print(f"Train batch size: {config['train_batch_size']}")
    print(f"Val batch size: {config['val_batch_size']}")

    # Create model
    model = Sd_model_lora(preprocessor=processor, **config)

    # train model
    model.train()

    # Save config in output folder
    shutil.copy("config/json/config.json", config["output_dir"])
