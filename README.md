# Fine tuning Stable Diffusion for Audio Spectrograms
## Dissertation project for MSc. Computing at Imperial College London

### Author: Ryan Perkins
#### Supervisors: Björn Schuller, Rodrigo Mira

**Abstract**: Stable Diffusion (SD) is a popular open-source model pre-trained on billions of image-text pairs, designed to generate images from textual prompts. With its extensive ecosystem and supportive community, SD offers functionalities like inpainting and outpainting along with a multitude of other features enabling enhanced control over generation. Notably, SD can be fine-tuned on consumer-level hardware, enabling users to adapt the model to their specific needs.

While diffusion models have been foundational in text-to-audio (TTA) systems, training them from scratch remains prohibitively expensive for many consumers. Moreover, these models may lack the functionality, support and tools that the SD ecosystem provides.

In *Stable Diffusion for Audio Spectrograms*, we pioneer the exploration of fine-tuning SD to generate non-music audio spectrograms. A primary aim is to democratise consumer access to fine-tuning their own text-to-audio systems and to harness the capabilities of the SD ecosystem for audio generation.

Through extensive empirical experiments, we find that while SD showed proficiency in generating certain audio classes, it faced challenges in others, suggesting a potential need for a more comprehensive training dataset. We also highlight the versatility of the SD ecosystem for both researchers and users.

Although our model did not surpass the current state-of-the-art, the insights emphasise the potential of SD in audio generation, its prospective role as a consumer-accessible TTA system, and avenues for further research.

We conclude by addressing ethical considerations, including potential misuse and copyright concerns, emphasising the importance of responsible advancement in this domain.

Audio samples illustrating our model's capabilities can be accessed [here/link].

## Overview of codebase
1. **config**: Contains a .json file used to specify training parameters and one utility function used for parsing this file.
2. **data**: Contains functions used for downloading and preprocessing our data, including spectrogram conversion code taken from [Riffusion](https://github.com/riffusion/riffusion). 
Also contains .csv used to download data (including files failed to download) as well as metadata files detailing samples used in our training.
3. **evaluation**: Contains evaluation code taken from [AudioLDM](https://github.com/haoheliu/audioldm_eval/tree/main/audioldm_eval) used for metric generation and code for generating required test data needed for this evaluation.
Also contains model checkpoints, high-level experiment analysis and outputs & utility functions related to our experiments with Web UI. 
Utility function convert_to_safetensors.py taken from [here](https://github.com/harrywang/finetune-sd/blob/main/convert-to-safetensors.py).
4. **model**: Contains training script adapted from [Diffusers](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py).
