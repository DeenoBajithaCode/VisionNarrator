# VisionNarrator - Image Caption Generator

This project generates captions for images using CLIP for image understanding and GPT-2 for text generation. It uses a simple Django API to upload an image and get a caption as a response.

## Features

- Upload an image and get a generated caption
- Uses pre-trained CLIP and GPT-2 models
- Flask API for easy testing (e.g., with Postman)

## Requirements

- Python 3.8+
- Transformers
- Django
- Pillow
- torch

Install dependencies:

```bash
pip install -r requirements.txt
