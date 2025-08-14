#!/usr/bin/env python

import torch
from PIL import Image
import os
import glob
import argparse

# Argument parsing
parser = argparse.ArgumentParser(description='Detect NSFW images using BLIP')
parser.add_argument('directory', type=str, help="dir with images")
parser.add_argument('-d', '--display', action='store_true', help="show image with label in title")
parser.add_argument('-g', '--gpu', action='store_true', help="use gpu")
parser.add_argument('-c', '--clean', action='store_true', help="print clean files (shows you the progress)")
args = parser.parse_args()

print(f"Searching in dir: {args.directory}")

if args.display:
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

# Recursive image search
image_files = []
for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
    image_files.extend(glob.glob(os.path.join(args.directory, "**", ext), recursive=True))

if not image_files:
    print("No images found!")
    exit()

# NSFW keywords (simple example)
nsfw_keywords = [
    "nude", "naked", "sexual", "porn", "breast", "penis", "vagina",
    "buttocks", "sex", "erotic", "explicit", "intimate"
]

# Process images
for filepath in image_files:
    try:
        image = Image.open(filepath).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)

        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

        if args.clean:
            print(f"clean: {filepath}")

        # Check if caption contains NSFW words
        if any(word in caption.lower() for word in nsfw_keywords):
            print(f"NSFW detected: {filepath} → {caption}")

            if args.display:
                plt.figure()
                plt.title(caption)
                plt.imshow(io.imread(filepath))
                plt.axis("off")
                plt.show()

    except Exception as e:
        print(f"Skipping {filepath}: {e}")

