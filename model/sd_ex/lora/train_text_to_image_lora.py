# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fine-tuning script for Stable Diffusion for text2image with support for LoRA."""

import json
import logging
import math
import os
import random
import shutil
import sys
from pathlib import Path

import datasets
import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchaudio
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

sys.path.append("../../../data")
from torchvision.transforms.functional import to_pil_image

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.18.0.dev0")

logger = get_logger(__name__, log_level="INFO")


class Sd_model_lora:
    def __init__(
        self,
        preprocessor,
        pretrained_model_name_or_path: str = None,
        revision: str = None,
        dataset_name: str = None,
        dataset_config_name: str = None,
        train_data_dir: str = None,
        val_data_dir: str = None,
        image_column: str = "image",
        caption_column: str = "text",
        validation_prompts: list[str] = None,
        num_validation_images: int = 4,
        validation_epochs: int = 1,
        max_train_samples: int = None,
        max_val_samples: int = None,
        output_dir: str = "sd-model-finetuned-lora",
        cache_dir: str = None,
        seed: int = None,
        resolution: int = 512,
        center_crop: bool = False,
        random_flip: bool = False,
        train_batch_size: int = 16,
        val_batch_size: int = 16,
        num_train_epochs: int = 100,
        max_train_steps: int = None,
        gradient_accumulation_steps: int = 1,
        gradient_checkpointing: bool = False,
        learning_rate: float = 1e-4,
        scale_lr: bool = False,
        lr_scheduler: str = "constant",
        lr_warmup_steps: int = 500,
        snr_gamma: float = None,
        use_8bit_adam: bool = False,
        allow_tf32: bool = False,
        dataloader_num_workers: int = 0,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_weight_decay: float = 1e-2,
        adam_epsilon: float = 1e-08,
        max_grad_norm: float = 1.0,
        push_to_hub: bool = False,
        hub_token: str = None,
        prediction_type: str = None,
        hub_model_id: str = None,
        logging_dir: str = "logs",
        mixed_precision: str = None,
        report_to: str = "tensorboard",
        local_rank: int = -1,
        checkpointing_steps: int = 500,
        checkpoints_total_limit: int = None,
        resume_from_checkpoint: str = None,
        enable_xformers_memory_efficient_attention: bool = False,
        noise_offset: float = 0,
    ):
        """
        Initialize a model with specified data, preprocessor, and hyperparameters.

        :param preprocessor: The preprocessor for data preparation.
        :param pretrained_model_name_or_path: Path to the pretrained model or its name.
        :param revision: Model revision.
        :param dataset_name: Name of the dataset.
        :param dataset_config_name: Configuration name of the dataset.
        :param train_data_dir: Path to training data directory.
        :param val_data_dir: Path to validation data directory.
        :param image_column: Column name for image data.
        :param caption_column: Column name for caption text.
        :param validation_prompts: Validation prompts.
        :param num_validation_images: Number of validation images.
        :param validation_epochs: Number of validation epochs.
        :param max_train_samples: Maximum number of training samples.
        :param max_val_samples: Maximum number of validation samples.
        :param output_dir: Path to output directory.
        :param cache_dir: Path to cache directory.
        :param seed: Random seed.
        :param resolution: Image resolution.
        :param center_crop: Apply center cropping if True.
        :param random_flip: Apply random flip if True.
        :param train_batch_size: Batch size for training.
        :param val_batch_size: Batch size for validation.
        :param num_train_epochs: Number of training epochs.
        :param max_train_steps: Maximum number of training steps.
        :param gradient_accumulation_steps: Number of steps for gradient accumulation.
        :param gradient_checkpointing: Enable gradient checkpointing if True.
        :param learning_rate: Learning rate for training.
        :param scale_lr: Scale learning rate if True.
        :param lr_scheduler: Learning rate scheduler type.
        :param lr_warmup_steps: Number of learning rate warmup steps.
        :param snr_gamma: SNR gamma value.
        :param use_8bit_adam: Use 8-bit Adam optimizer if True.
        :param allow_tf32: Allow TensorFlow 32 if True.
        :param dataloader_num_workers: Number of workers for data loading.
        :param adam_beta1: Beta1 parameter for Adam optimizer.
        :param adam_beta2: Beta2 parameter for Adam optimizer.
        :param adam_weight_decay: Weight decay for Adam optimizer.
        :param adam_epsilon: Epsilon value for Adam optimizer.
        :param max_grad_norm: Maximum gradient norm.
        :param push_to_hub: Push to hub if True.
        :param hub_token: Hub authentication token.
        :param prediction_type: Prediction type for the model.
        :param hub_model_id: ID of the hub model.
        :param logging_dir: Directory for logging.
        :param mixed_precision: Mixed precision mode.
        :param report_to: Reporting tool (e.g., TensorBoard).
        :param local_rank: Local rank for distributed training.
        :param checkpointing_steps: Number of steps for checkpointing.
        :param checkpoints_total_limit: Total limit for checkpoints.
        :param resume_from_checkpoint: Path to resume from a checkpoint.
        :param enable_xformers_memory_efficient_attention: Enable memory-efficient attention in transformers if True.
        :param noise_offset: Noise offset value.
        """

        self._preprocessor = preprocessor
        self._pretrained_model_name_or_path = pretrained_model_name_or_path
        self._seed = seed
        self._num_train_epochs = num_train_epochs
        self._max_train_steps = max_train_steps
        self._revision = revision
        self._dataset_name = dataset_name
        self._dataset_config = dataset_config_name
        self._image_column = image_column
        self._caption_column = caption_column
        self._validation_prompts = validation_prompts
        self._num_validation_images = num_validation_images
        self._train_data_dir = train_data_dir
        self._val_data_dir = val_data_dir
        self._validation_epochs = validation_epochs
        self._max_train_samples = max_train_samples
        self._max_val_samples = max_val_samples
        self._output_dir = output_dir
        self._cache_dir = cache_dir
        self._resolution = resolution
        self._center_crop = center_crop
        self._random_flip = random_flip
        self._train_batch_size = train_batch_size
        self._val_batch_size = val_batch_size
        self._gradient_accumulation_steps = gradient_accumulation_steps
        self._gradient_checkpointing = gradient_checkpointing
        self._learning_rate = learning_rate
        self._scale_lr = scale_lr
        self._lr_scheduler = lr_scheduler
        self._lr_warmup_steps = lr_warmup_steps
        self._snr_gamma = snr_gamma
        self._use_8bit_adam = use_8bit_adam
        self._allow_tf32 = allow_tf32
        self._dataloader_num_workers = dataloader_num_workers
        self._adam_beta1 = adam_beta1
        self._adam_beta2 = adam_beta2
        self._adam_weight_decay = adam_weight_decay
        self._adam_epsilon = adam_epsilon
        self._max_grad_norm = max_grad_norm
        self._push_to_hub = push_to_hub
        self._hub_token = hub_token
        self._prediction_type = prediction_type
        self._hub_model_id = hub_model_id
        self._logging_dir = logging_dir
        self._mixed_precision = mixed_precision
        self._report_to = report_to
        self._local_rank = local_rank
        self._checkpointing_steps = checkpointing_steps
        self._checkpoints_total_limit = checkpoints_total_limit
        self._resume_from_checkpoint = resume_from_checkpoint
        self._enable_xformers_memory_efficient_attention = (
            enable_xformers_memory_efficient_attention
        )
        self._noise_offset = noise_offset
        self._config = {
            key[1:]: value
            for key, value in vars(self).items()
            if key.startswith("_") and key != "_preprocessor"
        }
        self._config["validation_prompts"] = json.dumps(self._validation_prompts)

    def save_model_card(
        self,
        repo_id: str,
        images=None,
        base_model=str,
        dataset_name=str,
        repo_folder=None,
    ):
        img_str = ""
        for i, image in enumerate(images):
            image.save(os.path.join(repo_folder, f"image_{i}.png"))
            img_str += f"![img_{i}](./image_{i}.png)\n"

        yaml = f"""
    ---
    license: creativeml-openrail-m
    base_model: {base_model}
    tags:
    - stable-diffusion
    - stable-diffusion-diffusers
    - text-to-image
    - diffusers
    - lora
    inference: true
    ---
        """
        model_card = f"""
    # LoRA text2image fine-tuning - {repo_id}
    These are LoRA adaption weights for {base_model}. The weights were fine-tuned on the {dataset_name} dataset. You can find some example images in the following. \n
    {img_str}
    """
        with open(os.path.join(repo_folder, "README.md"), "w") as f:
            f.write(yaml + model_card)

    def train(self):
        DATASET_NAME_MAPPING = {
            "dataset": ("image", "text", "audiofile"),
        }

        logging_dir = Path(self._output_dir, self._logging_dir)

        accelerator_project_config = ProjectConfiguration(
            project_dir=self._output_dir, logging_dir=logging_dir
        )

        accelerator = Accelerator(
            gradient_accumulation_steps=self._gradient_accumulation_steps,
            mixed_precision=self._mixed_precision,
            log_with=self._report_to,
            project_config=accelerator_project_config,
        )
        if self._report_to == "wandb":
            if not is_wandb_available():
                raise ImportError(
                    "Make sure to install wandb if you want to use it for logging during training."
                )
            import wandb

        # Make one log on every process with the configuration for debugging.
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(accelerator.state, main_process_only=False)
        if accelerator.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        # If passed along, set the training seed now.
        if self._seed is not None:
            set_seed(self._seed)

        # Handle the repository creation
        if accelerator.is_main_process:
            if self._output_dir is not None:
                os.makedirs(self._output_dir, exist_ok=True)

            if self._push_to_hub:
                repo_id = create_repo(
                    repo_id=self._hub_model_id or Path(self._output_dir).name,
                    exist_ok=True,
                    token=self._hub_token,
                ).repo_id
        # Load scheduler, tokenizer and models.
        noise_scheduler = DDPMScheduler.from_pretrained(
            self._pretrained_model_name_or_path, subfolder="scheduler"
        )
        tokenizer = CLIPTokenizer.from_pretrained(
            self._pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=self._revision,
        )
        text_encoder = CLIPTextModel.from_pretrained(
            self._pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=self._revision,
        )
        vae = AutoencoderKL.from_pretrained(
            self._pretrained_model_name_or_path,
            subfolder="vae",
            revision=self._revision,
        )
        unet = UNet2DConditionModel.from_pretrained(
            self._pretrained_model_name_or_path,
            subfolder="unet",
            revision=self._revision,
        )
        # freeze parameters of models to save more memory
        unet.requires_grad_(False)
        vae.requires_grad_(False)

        text_encoder.requires_grad_(False)

        # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        # Move unet, vae and text_encoder to device and cast to weight_dtype
        unet.to(accelerator.device, dtype=weight_dtype)
        vae.to(accelerator.device, dtype=weight_dtype)
        text_encoder.to(accelerator.device, dtype=weight_dtype)

        # now we will add new LoRA weights to the attention layers
        # It's important to realize here how many attention weights will be added and of which sizes
        # The sizes of the attention layers consist only of two different variables:
        # 1) - the "hidden_size", which is increased according to `unet.config.block_out_channels`.
        # 2) - the "cross attention size", which is set to `unet.config.cross_attention_dim`.

        # Let's first see how many attention processors we will have to set.
        # For Stable Diffusion, it should be equal to:
        # - down blocks (2x attention layers) * (2x transformer layers) * (3x down blocks) = 12
        # - mid blocks (2x attention layers) * (1x transformer layers) * (1x mid blocks) = 2
        # - up blocks (2x attention layers) * (3x transformer layers) * (3x down blocks) = 18
        # => 32 layers

        # Set correct lora layers
        lora_attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]

            lora_attn_procs[name] = LoRAAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
            )

        unet.set_attn_processor(lora_attn_procs)

        if self._enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                import xformers

                xformers_version = version.parse(xformers.__version__)
                if xformers_version == version.parse("0.0.16"):
                    logger.warn(
                        "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                    )
                unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError(
                    "xformers is not available. Make sure it is installed correctly"
                )

        def compute_snr(self, timesteps):
            """
            Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
            """
            alphas_cumprod = noise_scheduler.alphas_cumprod
            sqrt_alphas_cumprod = alphas_cumprod**0.5
            sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

            # Expand the tensors.
            # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
            sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[
                timesteps
            ].float()
            while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
                sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
            alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(
                device=timesteps.device
            )[timesteps].float()
            while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
                sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
            sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

            # Compute SNR.
            snr = (alpha / sigma) ** 2
            return snr

        lora_layers = AttnProcsLayers(unet.attn_processors)

        # Enable TF32 for faster training on Ampere GPUs,
        # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        if self._allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        if self._scale_lr:
            self._learning_rate = (
                self._learning_rate
                * self._gradient_accumulation_steps
                * self._train_batch_size
                * accelerator.num_processes
            )

        # Initialize the optimizer
        if self._use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
                )

            optimizer_cls = bnb.optim.AdamW8bit
        else:
            optimizer_cls = torch.optim.AdamW

        optimizer = optimizer_cls(
            lora_layers.parameters(),
            lr=self._learning_rate,
            betas=(self._adam_beta1, self._adam_beta2),
            weight_decay=self._adam_weight_decay,
            eps=self._adam_epsilon,
        )

        # Get the datasets: you can either provide your own training and evaluation files (see below)
        # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

        # In distributed training, the load_dataset function guarantees that only one local process can concurrently
        # download the dataset.
        if self._dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            dataset = load_dataset(
                self._dataset_name,
                self._dataset_config_name,
                cache_dir=self._cache_dir,
            )
        else:
            data_files = {}
            if self._train_data_dir is not None:
                data_files["train"] = os.path.join(self._train_data_dir, "**")
            if (
                self._val_data_dir is not None
            ):  # make sure to include an argument for your validation data directory
                data_files["validation"] = os.path.join(self._val_data_dir, "**")
            dataset = load_dataset(
                "imagefolder",
                data_files=data_files,
                cache_dir=self._cache_dir,
            )
            # See more about loading custom images at
            # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

        # Preprocessing the datasets.
        # We need to tokenize inputs and targets.
        column_names = dataset["train"].column_names

        # 6. Get the column names for input/target.
        dataset_columns = DATASET_NAME_MAPPING.get(self._dataset_name, None)
        if self._image_column is None:
            image_column = (
                dataset_columns[0] if dataset_columns is not None else column_names[0]
            )
        else:
            image_column = self._image_column
            if image_column not in column_names:
                raise ValueError(
                    f"--image_column' value '{self._image_column}' needs to be one of: {', '.join(column_names)}"
                )
        if self._caption_column is None:
            caption_column = (
                dataset_columns[1] if dataset_columns is not None else column_names[1]
            )
        else:
            caption_column = self._caption_column
            if caption_column not in column_names:
                raise ValueError(
                    f"--caption_column' value '{self._caption_column}' needs to be one of: {', '.join(column_names)}"
                )

        # Handling the audiofile column
        audiofile_column = "audiofile"  # adjust this to the actual name in your dataset
        if audiofile_column not in column_names:
            raise ValueError(
                f"'audiofile_column' value '{audiofile_column}' needs to be one of: {', '.join(column_names)}"
            )

        # Preprocessing the datasets.
        # We need to tokenize input captions and transform the images.
        def tokenize_captions(examples, is_train=True):
            captions = []
            for caption in examples[caption_column]:
                if isinstance(caption, str):
                    captions.append(caption)
                elif isinstance(caption, (list, np.ndarray)):
                    # take a random caption if there are multiple
                    captions.append(random.choice(caption) if is_train else caption[0])
                else:
                    raise ValueError(
                        f"Caption column `{caption_column}` should contain either strings or lists of strings."
                    )
            inputs = tokenizer(
                captions,
                max_length=tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            return inputs.input_ids

        # Preprocessing the datasets.
        train_transforms = transforms.Compose(
            [
                transforms.Resize(
                    self._resolution,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.CenterCrop(self._resolution)
                if self._center_crop
                else transforms.RandomCrop(self._resolution),
                transforms.RandomHorizontalFlip()
                if self._random_flip
                else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                # transforms.Normalize([0.5], [0.5]),
            ]
        )

        def preprocess_train(examples):
            images = [image.convert("RGB") for image in examples[image_column]]
            examples["pixel_values"] = [train_transforms(image) for image in images]
            examples["input_ids"] = tokenize_captions(examples)

            # load audio data
            audio_files = [
                os.path.splitext(image_path)[0] + ".wav"
                for image_path in examples[audiofile_column]
            ]
            examples["audio"] = [
                torchaudio.load(wav_path)[0] for wav_path in audio_files
            ]

            return examples

        def preprocess_val(examples):
            images = [image.convert("RGB") for image in examples[image_column]]
            examples["pixel_values"] = [train_transforms(image) for image in images]
            examples["input_ids"] = tokenize_captions(examples, is_train=False)

            # load audio data
            audio_files = [
                os.path.splitext(image_path)[0] + ".wav"
                for image_path in examples[audiofile_column]
            ]
            examples["audio"] = [
                torchaudio.load(wav_path)[0] for wav_path in audio_files
            ]

            return examples

        with accelerator.main_process_first():
            if self._max_train_samples is not None:
                dataset["train"] = (
                    dataset["train"]
                    .shuffle(seed=self._seed)
                    .select(range(self._max_train_samples))
                )
            # Set the training transforms
            train_dataset = dataset["train"].with_transform(preprocess_train)

            if self._max_val_samples is not None:
                dataset["validation"] = dataset["validation"].select(
                    range(self._max_val_samples)
                )
            # Set the validation transforms
            val_dataset = dataset["validation"].with_transform(preprocess_val)

        def collate_fn(examples):
            pixel_values = torch.stack(
                [example["pixel_values"] for example in examples]
            )
            pixel_values = pixel_values.to(
                memory_format=torch.contiguous_format
            ).float()
            input_ids = torch.stack([example["input_ids"] for example in examples])

            # Adding audio into batch
            audio = torch.stack([example["audio"] for example in examples])

            return {
                "pixel_values": pixel_values,
                "input_ids": input_ids,
                "audio": audio,
            }

        # DataLoaders creation:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=collate_fn,
            batch_size=self._train_batch_size,
            num_workers=self._dataloader_num_workers,
        )

        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            shuffle=False,
            collate_fn=collate_fn,
            batch_size=self._val_batch_size,
            num_workers=self._dataloader_num_workers,
        )

        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / self._gradient_accumulation_steps
        )
        if self._max_train_steps is None:
            self._max_train_steps = self._num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True

        lr_scheduler = get_scheduler(
            self._lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self._lr_warmup_steps * self._gradient_accumulation_steps,
            num_training_steps=self._max_train_steps
            * self._gradient_accumulation_steps,
        )

        # Prepare everything with our `accelerator`.
        (
            lora_layers,
            optimizer,
            train_dataloader,
            val_dataloader,
            lr_scheduler,
        ) = accelerator.prepare(
            lora_layers, optimizer, train_dataloader, val_dataloader, lr_scheduler
        )

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / self._gradient_accumulation_steps
        )
        if overrode_max_train_steps:
            self._max_train_steps = self._num_train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        self._num_train_epochs = math.ceil(
            self._max_train_steps / num_update_steps_per_epoch
        )

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if accelerator.is_main_process:
            accelerator.init_trackers("sd_speech", config=(self._config))

        # Train!
        total_batch_size = (
            self._train_batch_size
            * accelerator.num_processes
            * self._gradient_accumulation_steps
        )

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {self._num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {self._train_batch_size}")
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
        )
        logger.info(
            f"  Gradient Accumulation steps = {self._gradient_accumulation_steps}"
        )
        logger.info(f"  Total optimization steps = {self._max_train_steps}")
        global_step = 0
        first_epoch = 0
        val_step = 0

        # Potentially load in the weights and states from a previous save
        if self._resume_from_checkpoint:
            if self._resume_from_checkpoint != "latest":
                path = os.path.basename(self._resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                dirs = os.listdir(self._output_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None

            if path is None:
                accelerator.print(
                    f"Checkpoint '{self._resume_from_checkpoint}' does not exist. Starting a new training run."
                )
                self._resume_from_checkpoint = None
            else:
                accelerator.print(f"Resuming from checkpoint {path}")
                accelerator.load_state(os.path.join(self._output_dir, path))
                global_step = int(path.split("-")[1])

                resume_global_step = global_step * self._gradient_accumulation_steps
                first_epoch = global_step // num_update_steps_per_epoch
                resume_step = resume_global_step % (
                    num_update_steps_per_epoch * self._gradient_accumulation_steps
                )

        # Only show the progress bar once on each machine.
        progress_bar = tqdm(
            range(global_step, self._max_train_steps),
            disable=not accelerator.is_local_main_process,
        )
        progress_bar.set_description("Steps")

        min_val = 99999999  # used for checkpointing

        for epoch in range(first_epoch, self._num_train_epochs):
            logger.info(f"Starting epoch {epoch}")

            # TRAINING LOOP
            unet.train()
            train_loss = 0.0

            for step, batch in enumerate(train_dataloader):
                # Skip steps until we reach the resumed step
                if (
                    self._resume_from_checkpoint
                    and epoch == first_epoch
                    and step < resume_step
                ):
                    if step % self._gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                    continue

                # logger.info(f"Starting training step {step}, global step {global_step}")
                pixel_values = batch["pixel_values"]
                image = to_pil_image(pixel_values[0])
                audio_data = batch["audio"]
                audio_data = audio_data.squeeze()
                if audio_data.dim() > 1 and audio_data.shape[0] > 1:
                    audio_data = audio_data.mean(dim=0)
                else:
                    audio_data = audio_data

                # Log the batch of training images and audio every 10 epochs
                if epoch == 0 or epoch % 10 == 0:
                    # Collect data to log after loop
                    wandb.log(
                        {
                            "train_input/training_images": wandb.Image(
                                batch["pixel_values"],
                                caption=f"Epoch {epoch} Step {step}",
                            )
                        },
                        commit=False,
                    )
                    wandb.log(
                        {
                            "train_input/training_audio": wandb.Audio(
                                audio_data.cpu().numpy(),
                                sample_rate=44100,
                                caption=f"Epoch {epoch} Step {step}",
                            )
                        },
                        commit=False,
                    )
                    wandb.log(
                        {
                            "train_input/training_images_audio (LOSSY)": wandb.Audio(
                                self._preprocessor.spec_to_wav_np(image),
                                sample_rate=44100,
                                caption=f"Epoch {epoch} Step {step}",
                            )
                        },
                        commit=False,
                    )

                with accelerator.accumulate(unet):
                    # Convert images to latent space
                    latents = vae.encode(
                        batch["pixel_values"].to(dtype=weight_dtype)
                    ).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    if self._noise_offset:
                        # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                        noise += self._noise_offset * torch.randn(
                            (latents.shape[0], latents.shape[1], 1, 1),
                            device=latents.device,
                        )

                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(
                        0,
                        noise_scheduler.config.num_train_timesteps,
                        (bsz,),
                        device=latents.device,
                    )
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    # Get the text embedding for conditioning
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                    # Get the target for loss depending on the prediction type
                    if self._prediction_type is not None:
                        # set prediction_type of scheduler if defined
                        noise_scheduler.register_to_config(
                            prediction_type=self._prediction_type
                        )

                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(
                            f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                        )

                    # Predict the noise residual and compute loss
                    model_pred = unet(
                        noisy_latents, timesteps, encoder_hidden_states
                    ).sample

                    if self._snr_gamma is None:
                        loss = F.mse_loss(
                            model_pred.float(), target.float(), reduction="mean"
                        )
                    else:
                        # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                        # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                        # This is discussed in Section 4.2 of the same paper.
                        snr = compute_snr(timesteps)
                        mse_loss_weights = (
                            torch.stack(
                                [snr, self._snr_gamma * torch.ones_like(timesteps)],
                                dim=1,
                            ).min(dim=1)[0]
                            / snr
                        )
                        # We first calculate the original loss. Then we mean over the non-batch dimensions and
                        # rebalance the sample-wise losses with their respective loss weights.
                        # Finally, we take the mean of the rebalanced loss.
                        loss = F.mse_loss(
                            model_pred.float(), target.float(), reduction="none"
                        )
                        loss = (
                            loss.mean(dim=list(range(1, len(loss.shape))))
                            * mse_loss_weights
                        )
                        loss = loss.mean()

                    # Gather the losses across all processes for logging (if we use distributed training).
                    avg_loss = accelerator.gather(
                        loss.repeat(self._train_batch_size)
                    ).mean()
                    avg_loss_per_step = avg_loss.item()
                    train_loss += avg_loss.item()

                    accelerator.log(
                        {"training_step_loss": avg_loss_per_step}, step=global_step
                    )

                    logger.info(f"train loss is {train_loss}")

                    # Backpropagate
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        params_to_clip = lora_layers.parameters()
                        accelerator.clip_grad_norm_(params_to_clip, self._max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    # if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    if global_step % self._checkpointing_steps == 0:
                        if accelerator.is_main_process:
                            # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                            if self._checkpoints_total_limit is not None:
                                checkpoints = os.listdir(self._output_dir)
                                checkpoints = [
                                    d for d in checkpoints if d.startswith("checkpoint")
                                ]
                                checkpoints = sorted(
                                    checkpoints, key=lambda x: int(x.split("-")[1])
                                )

                                # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                                if len(checkpoints) >= self._checkpoints_total_limit:
                                    num_to_remove = (
                                        len(checkpoints)
                                        - self._checkpoints_total_limit
                                        + 1
                                    )
                                    removing_checkpoints = checkpoints[0:num_to_remove]

                                    logger.info(
                                        f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                    )
                                    logger.info(
                                        f"removing checkpoints: {', '.join(removing_checkpoints)}"
                                    )

                                    for removing_checkpoint in removing_checkpoints:
                                        removing_checkpoint = os.path.join(
                                            self._output_dir, removing_checkpoint
                                        )
                                        shutil.rmtree(removing_checkpoint)

                            save_path = os.path.join(
                                self._output_dir, f"checkpoint-{global_step}"
                            )
                            accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}")

                logs = {
                    "step_loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                progress_bar.set_postfix(**logs)

            avg_train_loss_per_epoch = train_loss / len(train_dataloader)
            accelerator.log(
                {"avg_train_loss_per_epoch": avg_train_loss_per_epoch}, step=global_step
            )

            # reset training loss for next epoch
            train_loss = 0.0

            # VALIDATION LOOP
            if epoch % self._validation_epochs == 0:
                unet.eval()  # set model to evaluation mode
                valid_loss = 0.0
                val_images_log = []
                val_audio_log = []
                val_images_audio_log = []

                for step, batch in enumerate(val_dataloader):
                    val_step += 1

                    # logger.info(f"Starting val step {step}, global step {global_step}")
                    pixel_values = batch["pixel_values"]
                    image = to_pil_image(pixel_values[0])

                    # Ensuring audio data has correct dimensions for logging
                    audio_data = batch["audio"]
                    audio_data = audio_data.squeeze()
                    if audio_data.dim() > 1 and audio_data.shape[0] > 1:
                        audio_data = audio_data.mean(dim=0)
                    else:
                        audio_data = audio_data

                    # Log random validation data
                    val_images_log.append(
                        wandb.Image(
                            batch["pixel_values"], caption=f"Epoch {epoch} Step {step}"
                        )
                    )
                    val_audio_log.append(
                        wandb.Audio(
                            audio_data.cpu().numpy(),
                            sample_rate=44100,
                            caption=f"Epoch {epoch} Step {step}",
                        )
                    )
                    val_images_audio_log.append(
                        wandb.Audio(
                            self._preprocessor.spec_to_wav_np(image),
                            sample_rate=44100,
                            caption=f"Epoch {epoch} Step {step}",
                        )
                    )

                    with torch.no_grad():  # disable gradient calculation
                        latents = vae.encode(
                            batch["pixel_values"].to(dtype=weight_dtype)
                        ).latent_dist.sample()
                        latents = latents * vae.config.scaling_factor

                        noise = torch.randn_like(latents)

                        bsz = latents.shape[0]
                        timesteps = torch.randint(
                            0,
                            noise_scheduler.config.num_train_timesteps,
                            (bsz,),
                            device=latents.device,
                        )
                        timesteps = timesteps.long()

                        noisy_latents = noise_scheduler.add_noise(
                            latents, noise, timesteps
                        )

                        encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                        if noise_scheduler.config.prediction_type == "epsilon":
                            target = noise
                        elif noise_scheduler.config.prediction_type == "v_prediction":
                            target = noise_scheduler.get_velocity(
                                latents, noise, timesteps
                            )
                        else:
                            raise ValueError(
                                f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                            )

                        model_pred = unet(
                            noisy_latents, timesteps, encoder_hidden_states
                        ).sample

                        if self._snr_gamma is None:
                            val_loss = F.mse_loss(
                                model_pred.float(), target.float(), reduction="mean"
                            )
                        else:
                            snr = compute_snr(timesteps)
                            mse_loss_weights = (
                                torch.stack(
                                    [snr, self._snr_gamma * torch.ones_like(timesteps)],
                                    dim=1,
                                ).min(dim=1)[0]
                                / snr
                            )
                            val_loss = F.mse_loss(
                                model_pred.float(), target.float(), reduction="none"
                            )
                            val_loss = (
                                val_loss.mean(dim=list(range(1, len(val_loss.shape))))
                                * mse_loss_weights
                            )
                            val_loss = val_loss.mean()

                        val_avg_loss = accelerator.gather(
                            val_loss.repeat(self._val_batch_size)
                        ).mean()
                        avg_val_loss_per_step = (
                            val_avg_loss.item()
                        )  # calculate average loss per step
                        valid_loss += (
                            val_avg_loss.item()
                        )  # accumulate average loss over all steps

                        logger.info(
                            f"Per validation step average loss is {avg_val_loss_per_step}"
                        )
                        logger.info(
                            f"Cumulative validation average loss is {valid_loss}"
                        )

                # At the end of each epoch, randomly select one sample to log
                random_index = torch.randint(high=len(val_images_log), size=(1,)).item()

                # Extract the selected image and audio
                selected_image = val_images_log[random_index]
                selected_audio = val_audio_log[random_index]
                selected_image_audio = val_images_audio_log[random_index]

                wandb.log({"val_input/validation_images": selected_image}, commit=False)
                wandb.log({"val_input/validation_audio": selected_audio}, commit=False)
                wandb.log(
                    {"val_input/validation_images_audio (LOSSY)": selected_image_audio},
                    commit=False,
                )

                avg_valid_loss_per_epoch = valid_loss / len(
                    val_dataloader
                )  # calculate average validation loss per epoch
                logger.info(
                    f"Average validation loss for Epoch {epoch} is {avg_valid_loss_per_epoch}"
                )
                accelerator.log(
                    {"avg_valid_loss_per_epoch": avg_valid_loss_per_epoch},
                    step=global_step,
                )  # log the average validation loss per epoch

                # Save best performing checkpoint
                if avg_valid_loss_per_epoch <= min_val:
                    min_val = avg_valid_loss_per_epoch
                    save_path = os.path.join(
                        self._output_dir, f"checkpoint-{global_step}-best"
                    )

                    accelerator.save_state(save_path)

            if global_step >= self._max_train_steps:
                break
            if accelerator.is_main_process:
                if (
                    self._validation_prompts is not None
                    and epoch % self._validation_epochs == 0
                ):
                    logger.info(
                        f"Running validation... \n Generating {self._num_validation_images} images for each prompt in:"
                        f" {self._validation_prompts}."
                    )
                    # create pipeline
                    pipeline = DiffusionPipeline.from_pretrained(
                        self._pretrained_model_name_or_path,
                        unet=accelerator.unwrap_model(unet),
                        revision=self._revision,
                        torch_dtype=weight_dtype,
                    )
                    pipeline = pipeline.to(accelerator.device)
                    pipeline.set_progress_bar_config(disable=True)

                    # run inference
                    generator = torch.Generator(device=accelerator.device)
                    if self._seed is not None:
                        generator = generator.manual_seed(self._seed)

                    images = []
                    audios = []
                    image_prompts = []
                    audio_prompts = []

                    for prompt in self._validation_prompts:
                        for _ in range(self._num_validation_images):
                            img = pipeline(
                                prompt, num_inference_steps=30, generator=generator
                            ).images[0]
                            audio = self._preprocessor.spec_to_wav_np(img)

                            images.append(img)
                            audios.append(audio)
                            image_prompts.append(prompt)
                            audio_prompts.append(prompt)

                    image_logs = [
                        wandb.Image(image, caption=f"{i}: {image_prompts[i]}")
                        for i, image in enumerate(images)
                    ]
                    audio_logs = [
                        wandb.Audio(
                            audio, sample_rate=44100, caption=f"{i}: {audio_prompts[i]}"
                        )
                        for i, audio in enumerate(audios)
                    ]

                    wandb.log(
                        {
                            "val_inf/validation_inference_image": image_logs,
                            "val_inf/validation_inference_audio": audio_logs,
                        },
                        commit=False,
                    )

                    del pipeline
                    torch.cuda.empty_cache()

        # Save final model
        save_path = os.path.join(self._output_dir, f"checkpoint-{global_step}-final")

        accelerator.save_state(save_path)

        # Save the lora layers
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unet = unet.to(torch.float32)
            unet.save_attn_procs(self._output_dir)

            if self._push_to_hub:
                self.save_model_card(
                    repo_id,
                    images=images,
                    base_model=self._pretrained_model_name_or_path,
                    dataset_name=self._dataset_name,
                    repo_folder=self._output_dir,
                )
                upload_folder(
                    repo_id=repo_id,
                    folder_path=self._output_dir,
                    commit_message="End of training",
                    ignore_patterns=["step_*", "epoch_*"],
                )

        # Final inference
        # Load previous pipeline
        pipeline = DiffusionPipeline.from_pretrained(
            self._pretrained_model_name_or_path,
            revision=self._revision,
            torch_dtype=weight_dtype,
        )
        pipeline = pipeline.to(accelerator.device)

        # load attention processors
        pipeline.unet.load_attn_procs(self._output_dir)

        # run inference
        generator = torch.Generator(device=accelerator.device)
        if self._seed is not None:
            generator = generator.manual_seed(self._seed)

        images = []
        image_captions = []

        for prompt in self._validation_prompts:
            for _ in range(self._num_validation_images):
                img = pipeline(
                    prompt, num_inference_steps=30, generator=generator
                ).images[0]
                images.append(img)
                image_captions.append(prompt)

        if accelerator.is_main_process:
            image_logs = [
                wandb.Image(image, caption=caption)
                for image, caption in zip(images, image_captions)
            ]
            audio_logs = [
                wandb.Audio(
                    self._preprocessor.spec_to_wav_np(image),
                    sample_rate=44100,
                    caption=caption,
                )
                for image, caption in zip(images, image_captions)
            ]
            wandb.log(
                {
                    "test/test_image": image_logs,
                    "test/test_audio": audio_logs,
                }
            )

        accelerator.end_training()
