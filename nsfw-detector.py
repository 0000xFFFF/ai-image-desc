#!/usr/bin/env python

import torch
from PIL import Image
import os
import glob
import argparse
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Argument parsing
parser = argparse.ArgumentParser(description='Detect NSFW images using BLIP (auto moderation)')
parser.add_argument('directory', type=str, help="dir with images")
parser.add_argument('-s', '--show', action='store_true', help="show images (clean+nsfw) with label in title")
parser.add_argument('-sc', '--show_clean', action='store_true', help="show clean images with label in title")
parser.add_argument('-sn', '--show_nsfw', action='store_true', help="show nsfw images with label in title")
parser.add_argument('-g', '--gpu', action='store_true', help="use gpu")
parser.add_argument('-c', '--clean', action='store_true', help="print clean files (shows progress)")
parser.add_argument('-b', '--batch', type=int, default=8, help="batch size for GPU processing")
args = parser.parse_args()

if args.show or args.show_clean or args.show_nsfw:
    import matplotlib.pyplot as plt
    from skimage import io

def show(image_path, caption):
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
model.eval()

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

# Threaded image loading
def load_image(path):
    try:
        img = Image.open(path).convert("RGB")
        return path, img
    except Exception as e:
        return path, None

with ThreadPoolExecutor(max_workers=8) as executor:
    loaded_images = list(tqdm(executor.map(load_image, image_files), total=len(image_files), desc="Loading images"))

# Filter out failed loads
loaded_images = [(p, img) for p, img in loaded_images if img is not None]

# Results storage
nsfw_results = []
clean_results = []

# Batch processing with mixed precision
batch_size = args.batch
print(f"Batch size: {batch_size}")
for i in tqdm(range(0, len(loaded_images), batch_size), desc="Processing images"):
    batch = loaded_images[i:i+batch_size]
    paths, images = zip(*batch)

    inputs = processor(images=images, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=(device=="cuda")):
            outputs = model.generate(**inputs)

    captions = [processor.decode(o, skip_special_tokens=True) for o in outputs]

    for path, caption in zip(paths, captions):
        if any(word in caption.lower() for word in nsfw_keywords):
            nsfw_results.append((path, caption))
        else:
            clean_results.append((path, caption))
            if args.clean:
                print(f"clean: {path}")

# Summary output
print("\n===== SUMMARY =====")
print(f"Total images processed: {len(loaded_images)}")
print(f"NSFW detected: {len(nsfw_results)}")
print(f"Clean: {len(clean_results)}")

if nsfw_results:
    print("\nNSFW Images:")
    for path, caption in nsfw_results:
        print(f"{path} → {caption}")
        if args.show or args.show_nsfw:
            show(path, caption)

if clean_results:
    print("\nClean Images:")
    for path, caption in clean_results:
        print(f"{path} → {caption}")
        if args.show or args.show_clean:
            show(path, caption)

