#!/usr/bin/env python

import torch
from PIL import Image
import os
import glob
import argparse

# Argument parsing
parser = argparse.ArgumentParser(description='Detect NSFW images using BLIP (auto modding)')
parser.add_argument('directory', type=str, help="dir with images")
parser.add_argument('-s', '--show', action='store_true', help="show images (clean+nsfw) with label in title")
parser.add_argument('-dc', '--show_clean', action='store_true', help="show clean images with label in title")
parser.add_argument('-dn', '--show_nsfw', action='store_true', help="show nsfw images with label in title")
parser.add_argument('-g', '--gpu', action='store_true', help="use gpu")
parser.add_argument('-c', '--clean', action='store_true', help="print clean files (shows you the progress)")
args = parser.parse_args()

if args.show:
    import matplotlib.pyplot as plt
    from skimage import io

def show(image_path):
    plt.figure()
    plt.title(caption)
    plt.imshow(io.imread(image_path))
    plt.axis("off")
    plt.show()

print(f"Searching in dir: {args.directory}")

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

# NSFW keywords
nsfw_keywords = [
    "nude", "naked", "sexual", "porn", "breast", "penis", "vagina",
    "buttocks", "sex", "erotic", "explicit", "intimate"
]

# Results storage
nsfw_results = []
clean_results = []

# Process images
for filepath in image_files:
    try:
        image = Image.open(filepath).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)

        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

        # Check if caption contains NSFW words
        if any(word in caption.lower() for word in nsfw_keywords):
            nsfw_results.append((filepath, caption))
        else:
            clean_results.append((filepath, caption))
            if args.clean:
                print(f"clean: {filepath}")

    except Exception as e:
        print(f"Skipping {filepath}: {e}")

# Summary output
print("\n===== SUMMARY =====")
print(f"Total images processed: {len(image_files)}")
print(f"NSFW detected: {len(nsfw_results)}")
print(f"Clean: {len(clean_results)}")

if nsfw_results:
    print("\nNSFW Images:")
    for path, caption in nsfw_results:
        print(f"{path} → {caption}")
        if args.show or args.show_nsfw:
            show(path)

if clean_results:
    print("\nClean Images:")
    for path, caption in clean_results:
        print(f"{path} → {caption}")
        if args.show or args.show_clean:
            show(path)

