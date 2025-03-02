# Stable Diffusion Image Generation

## Overview
This project utilizes Stable Diffusion 2.1 to generate AI-powered images based on textual prompts. The implementation is done using the `diffusers` library from Hugging Face and runs on a CUDA-compatible GPU for efficient processing.

## Installation
To set up the project, install the required dependencies:
```bash
pip install torch torchvision torchaudio xformers scipy diffusers matplotlib
```

## Model Setup
The script loads the Stable Diffusion 2.1 model from `stabilityai/stable-diffusion-2-1` and applies the `DPMSolverMultistepScheduler` for improved image generation.

## Usage
Run the following Python script to generate an image:

```python
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import matplotlib.pyplot as plt

# Clear CUDA cache
torch.cuda.empty_cache()

# Load model
model_id = "stabilityai/stable-diffusion-2-1"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

# Generate image
prompt = "A cat with a red colour cap"
image = pipe(prompt, width=1000, height=1000).images[0]

# Display image
plt.imshow(image)
plt.axis('off')  # Hide axis numbers and ticks
plt.show()
```

## Features
- Uses `Stable Diffusion 2.1` for text-to-image generation.
- Runs on GPU (`cuda`) for efficient processing.
- Supports different prompts and image resolutions.

## Requirements
- Python 3.7+
- CUDA-compatible GPU
- PyTorch with CUDA support
- `diffusers` library

## License
This project is open-source and follows the license of the `stabilityai/stable-diffusion-2-1` model.

## Acknowledgments
Special thanks to Stability AI and Hugging Face for providing the model and tools for AI-powered image generation.

