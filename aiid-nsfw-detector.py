#!/usr/bin/env python
import torch
from PIL import Image
import os
import glob
import argparse
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import csv
import gc
import re
import string

# Argument parsing
parser = argparse.ArgumentParser(description='Detect NSFW images using BLIP (auto moderation)')
parser.add_argument('input', type=str, help="image file or directory with images")
parser.add_argument('-g', '--gpu', action='store_true', help="use GPU")
parser.add_argument('-c', '--count', metavar="number", type=int, default=1, help="how many captions to generate per image (default: 1)")
parser.add_argument('-a', '--all', action='store_true', help="output all generated captions (default: output only matched caption or last if no match)")
parser.add_argument('-s', '--show', action='store_true', help="show images (clean+nsfw) with caption in title after processing all")
parser.add_argument('-sc', '--show_clean', action='store_true', help="show clean images with caption in title after processing all")
parser.add_argument('-sn', '--show_nsfw', action='store_true', help="show nsfw images with caption in title after processing all")
parser.add_argument('-pc', '--print_clean', action='store_true', help="print clean images (default: don't print)")
parser.add_argument('-b', '--batch', metavar="number", type=int, default=8, help="batch size for GPU processing")
parser.add_argument('-lb', '--load_batch', metavar="number", type=int, default=256, help="batch size for loading images into memory")
parser.add_argument('-oc', '--output_clean', metavar="clean.csv", type=str, help="output clean image(s) with caption to CSV file")
parser.add_argument('-on', '--output_nsfw', metavar="nsfw.csv", type=str, help="output nsfw image(s) with caption CSV file")
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

# NSFW keywords
nsfw_keywords = set()
nsfw_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "nsfw.lst")
with open(nsfw_file, "r") as f:
    for line in f:
        x = line.strip()
        if x not in nsfw_keywords:
            nsfw_keywords.add(x)

print(f"Loaded {len(nsfw_keywords)} nsfw keywords")

# Collect image files
image_files = []
if os.path.isfile(args.input):
    image_files = [args.input]
elif os.path.isdir(args.input):
    print(f"Searching in dir: {args.input}")
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        image_files.extend(glob.glob(os.path.join(args.input, "**", ext), recursive=True))
    print(f"Found {len(image_files)} images")
else:
    print("Error: Input path is neither a file nor a directory")
    exit()

if not image_files:
    print("No images found!")
    exit()

# Device
device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print(f"Processing batch size: {args.batch}")
print(f"Loading batch size: {args.load_batch}")

# Load BLIP
from transformers import BlipProcessor, BlipForConditionalGeneration
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base",
    use_fast=False
)
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
model.to(device)
model.eval()

# Threaded image loading function
def load_image(path):
    try:
        img = Image.open(path).convert("RGB")
        return path, img
    except Exception:
        return path, None

# Results storage
nsfw_results = []
clean_results = []
total_processed = 0
failed_loads = 0

# Regex pattern for NSFW check
pattern = re.compile(r"\\b(" + "|".join(map(re.escape, nsfw_keywords)) + r")\\b", re.IGNORECASE)


# Multi-caption generation function
def generate_captions(img, num):
    """Generate multiple captions for a single image in one forward pass."""
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=40,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            temperature=0.7,
            num_return_sequences=num  # generate N captions at once
        )
    # Deduplicate captions
    captions = [processor.decode(o, skip_special_tokens=True) for o in outputs]
    return set(captions)


# Processing loop
load_batch_size = args.load_batch
processing_batch_size = args.batch

with tqdm(total=len(image_files), desc="Overall progress", position=0) as overall_pbar:
    with tqdm(total=0, desc="Current batch", position=1, leave=False) as batch_pbar:
        for load_start in range(0, len(image_files), load_batch_size):
            load_end = min(load_start + load_batch_size, len(image_files))
            current_batch_paths = image_files[load_start:load_end]

            batch_pbar.reset(total=len(current_batch_paths))
            batch_pbar.set_description(f"Loading batch {load_start//load_batch_size + 1}/{(len(image_files)-1)//load_batch_size + 1}")

            loaded_images = []
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                for result in executor.map(load_image, current_batch_paths):
                    loaded_images.append(result)
                    batch_pbar.update(1)

            valid_images = [(p, img) for p, img in loaded_images if img is not None]
            failed_loads += len(loaded_images) - len(valid_images)

            if not valid_images:
                overall_pbar.update(len(current_batch_paths))
                continue

            batch_pbar.reset(total=len(valid_images))
            batch_pbar.set_description(f"Processing batch {load_start//load_batch_size + 1}")

            for proc_start in range(0, len(valid_images), processing_batch_size):
                proc_end = min(proc_start + processing_batch_size, len(valid_images))
                batch = valid_images[proc_start:proc_end]
                paths, images = zip(*batch)

                for path, img in batch:
                    captions = generate_captions(img, args.count)

                    nsfw = False
                    for i, caption in enumerate(captions):
                        pattern = re.compile(r"\b(" + "|".join(map(re.escape, nsfw_keywords)) + r")\b", re.IGNORECASE)
                        text = caption.lower()
                        match = pattern.search(text)
                        if match:
                            nsfw = True
                            break

                    captions_str = caption
                    if args.all:
                        captions_str = " ; ".join(captions)

                    if nsfw:
                        nsfw_results.append((path, captions_str, f"{i} - {match.group(1)}"))
                    else:
                        clean_results.append((path, captions_str))

                total_processed += len(batch)
                batch_pbar.update(len(batch))
                overall_pbar.update(len(batch))

                if device == "cuda":
                    torch.cuda.empty_cache()

            del valid_images
            del loaded_images
            gc.collect()

            if device == "cuda":
                torch.cuda.empty_cache()


def save_csv(file, results, nsfw=False):
    try:
        with open(file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile, delimiter='|')
            if nsfw:
                writer.writerow(['Path', 'Caption(s)', 'MatchedWord'])
            else:
                writer.writerow(['Path', 'Caption(s)'])
            writer.writerows(results)
        print(f"Results saved to: {file}")
    except Exception as e:
        print(f"Failed to save CSV: {e}")

if nsfw_results:
    if args.output_nsfw:
        save_csv(args.output_nsfw, nsfw_results, nsfw=True)
    print("+==[ NSFW Images:")
    for path, captions, match in nsfw_results:
        print(f"|-- {path} → {captions} → {match}")
        if args.show or args.show_nsfw:
            show(path, captions)

if clean_results:
    if args.output_clean:
        save_csv(args.output_clean, clean_results)

    if args.print_clean or args.show or args.show_clean:
        print("+==[ Clean Images:")
        for path, captions in clean_results:
            print(f"|-- {path} → {captions}")
            if args.show or args.show_clean:
                show(path, captions)

# Summary
print(f"\n{"-"*20} SUMMARY {"-"*20}")
print(f"Total images found....: {len(image_files)}")
print(f"Total images processed: {total_processed}")
print(f"Failed to load........: {failed_loads}")
print(f"Clean images..........: {len(clean_results)}")
print(f"NSFW images...........: {len(nsfw_results)}")

