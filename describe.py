#!/usr/bin/env python

import torch
from PIL import Image
import os
import glob
import argparse

# Argument parsing
parser = argparse.ArgumentParser(description='Describe images in English using BLIP')
parser.add_argument('directory', type=str, help="dir with images")
parser.add_argument('-s', '--show', action='store_true', help="show image with label in title")
parser.add_argument('-g', '--gpu', action='store_true', help="use gpu")
args = parser.parse_args()

print(f"Searching in dir: {args.directory}")

if args.show:
    import matplotlib.pyplot as plt
    from skimage import io

# Device
device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load BLIP
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
model.to(device)

# Find images recursively
image_files = []
for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
    image_files.extend(glob.glob(os.path.join(args.directory, "**", ext), recursive=True))

if not image_files:
    print("No images found!")
    exit()

# Generate captions
for filepath in image_files:
    try:
        image = Image.open(filepath).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)

        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

        print(f"{filepath} → {caption}")

        if args.show:
            plt.figure()
            plt.title(caption)
            plt.imshow(io.imread(filepath))
            plt.axis("off")
            plt.show()

    except Exception as e:
        print(f"Skipping {filepath}: {e}")

