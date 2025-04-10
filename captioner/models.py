import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from PIL import Image
import clip
import numpy as np

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device)

# Load GPT-2 model and tokenizer
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model.to(device)

def generate_caption(image_path):
    # Process image with CLIP
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text = clip.tokenize(["a photo of a person"]).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        similarity = (image_features @ text_features.T).squeeze(0)
        caption = similarity.argmax().item()

    # Use GPT-2 to generate text from CLIP output
    input_text = f"Image caption: {caption}"
    inputs = gpt2_tokenizer.encode(input_text, return_tensors="pt").to(device)
    output = gpt2_model.generate(inputs, max_length=50, num_return_sequences=1)
    generated_caption = gpt2_tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_caption
