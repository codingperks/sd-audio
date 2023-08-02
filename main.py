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
    model = Sd_model_lora(
        preprocessor=processor,
        pretrained_model_name_or_path=config["pretrained_model_name_or_path"],
        revision=config["revision"],
        dataset_name=config["dataset_name"],
        dataset_config_name=config["dataset_config_name"],
        train_data_dir=config["train_data_dir"],
        val_data_dir=config["val_data_dir"],
        image_column=config["image_column"],
        caption_column=config["caption_column"],
        validation_prompts=config["validation_prompts"],
        num_validation_images=config["num_validation_images"],
        validation_epochs=config["validation_epochs"],
        max_train_samples=config["max_train_samples"],
        max_val_samples=config["max_val_samples"],
        output_dir=config["output_dir"],
        cache_dir=config["cache_dir"],
        seed=config["seed"],
        resolution=config["resolution"],
        center_crop=config["center_crop"],
        random_flip=config["random_flip"],
        train_batch_size=config["train_batch_size"],
        val_batch_size=config["val_batch_size"],
        num_train_epochs=config["num_train_epochs"],
        max_train_steps=config["max_train_steps"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        gradient_checkpointing=config["gradient_checkpointing"],
        learning_rate=config["learning_rate"],
        scale_lr=config["scale_lr"],
        lr_scheduler=config["lr_scheduler"],
        lr_warmup_steps=config["lr_warmup_steps"],
        snr_gamma=config["snr_gamma"],
        use_8bit_adam=config["use_8bit_adam"],
        allow_tf32=config["allow_tf32"],
        dataloader_num_workers=config["dataloader_num_workers"],
        adam_beta1=config["adam_beta1"],
        adam_beta2=config["adam_beta2"],
        adam_weight_decay=config["adam_weight_decay"],
        adam_epsilon=config["adam_epsilon"],
        max_grad_norm=config["max_grad_norm"],
        push_to_hub=config["push_to_hub"],
        hub_token=config["hub_token"],
        prediction_type=config["prediction_type"],
        hub_model_id=config["hub_model_id"],
        logging_dir=config["logging_dir"],
        mixed_precision=config["mixed_precision"],
        report_to=config["report_to"],
        local_rank=config["local_rank"],
        checkpointing_steps=config["checkpointing_steps"],
        checkpoints_total_limit=config["checkpoints_total_limit"],
        resume_from_checkpoint=config["resume_from_checkpoint"],
        enable_xformers_memory_efficient_attention=config[
            "enable_xformers_memory_efficient_attention"
        ],
        noise_offset=config["noise_offset"],
    )

    # train model
    model.train()
