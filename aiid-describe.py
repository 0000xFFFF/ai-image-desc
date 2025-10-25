#!/usr/bin/env python
import torch
from PIL import Image
import os
import glob
import argparse
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import gc
import csv

# Argument parsing
parser = argparse.ArgumentParser(description='Describe images in English using BLIP')
parser.add_argument('input', type=str, help="image file or directory with images")
parser.add_argument('-s', '--show', action='store_true', help="show image(s) with caption in title after processing all")
parser.add_argument('-c', '--count', metavar="number", type=int, default=1, help="how many captions to generate per image (default: 1)")
parser.add_argument('-w', '--words', action='store_true', help="turn captions into keywords that will be sorted")
parser.add_argument('-wc', '--words_clean', action='store_true', help="same as -w + remove function words like ('the', 'a', 'an', ...)")
parser.add_argument('-g', '--gpu', action='store_true', help="use gpu")
parser.add_argument('-b', '--batch', metavar="number", type=int, default=8, help="batch size for GPU processing")
parser.add_argument('-lb', '--load_batch', metavar="number", type=int, default=256, help="batch size for loading images into memory")
parser.add_argument('-o', '--output', metavar="file.csv", type=str, help="output results to CSV file")
parser.add_argument('-od', '--output_delimiter', metavar="file.csv", type=str, default="|", help="delimiter to use when outputing csv file (default: '|')")
parser.add_argument('-d', '--defaults', action='store_true', help="-c 5 -wc -g -b 128 -lb 256 -o output.csv")
parser.add_argument('-d2', '--defaults2', action='store_true', help="-c 5 -wc -g -b 256 -lb 512 -o output.csv")
args = parser.parse_args()

if args.defaults:
    args.count = 5
    args.words_clean = True
    args.gpu = True
    args.batch = 128
    args.load_batch = 256
    if not args.output:
        args.output = "output.csv"
        
if args.defaults2:
    args.count = 5
    args.words_clean = True
    args.gpu = True
    args.batch = 256
    args.load_batch = 512
    if not args.output:
        args.output = "output.csv"

if args.show:
    import matplotlib.pyplot as plt
    from skimage import io

if args.words_clean:
    args.words = True

print(args)


def show(image_path, caption):
    plt.figure()
    plt.title(caption)
    plt.imshow(io.imread(image_path))
    plt.axis("off")
    plt.show()

# Collect image files
image_files = []
if os.path.isfile(args.input):
    # Single file
    image_files = [args.input]
elif os.path.isdir(args.input):
    print(f"Searching in dir: {args.input}")
    # Directory (recursive search)
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
    except Exception as e:
        return path, None

# Results storage
results = []
total_processed = 0
failed_loads = 0

words_blacklist = set({"the", "a", "an", "at", "in", "of", "as", "to"})

specials = set({".", ",", "!", "?", ";", ":", "'", "\"",
                "(", ")", "[", "]", "{", "}", "-", "_",
                "/", "\\", "“", "”", "‘", "’", "—", "–", "…", "·", "•", "<", ">",
                "@", "#", "$", "%", "^", "&", "*", "+", "=", "`", "~", "|"})

def captions_to_words(captions):
    words = set()
    for i in captions:
        for w in i.split(" "):
            if args.words_clean:
                if w not in words and w not in words_blacklist:
                    for s in specials:
                        w = w.replace(s, "")
                    words.add(w.strip())
            else:
                if w not in words:
                    words.add(w)
    words_list = list(words)
    words_list.sort()
    return words_list


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
    if args.words:
        captions = captions_to_words(captions)

    return set(captions)


# Open CSV file if specified
csv_file = None
csv_writer = None
if args.output:
    try:
        csv_file = open(args.output, 'w', newline='', encoding='utf-8')
        csv_writer = csv.writer(csv_file, delimiter=args.output_delimiter)
        csv_writer.writerow(['Path', 'Caption'])  # Header
    except Exception as e:
        print(f"Failed to open CSV file for writing: {e}")
        # Disable output if file can't be opened
        args.output = None
        if csv_file:
            csv_file.close()


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

                batch_results = []

                if args.count == 1:
                    # Process batch with BLIP
                    inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
                    
                    with torch.no_grad():
                        with torch.amp.autocast("cuda", enabled=(device=="cuda")):
                            outputs = model.generate(**inputs)
                    
                    captions = [processor.decode(o, skip_special_tokens=True) for o in outputs]
                    
                    # Store results
                    for path, caption in zip(paths, captions):

                        caption_str = caption
                        if args.words:
                            caption_str = " ".join(captions_to_words(list(caption)))

                        batch_results.append((path, caption_str))

                else:
                    for path, img in batch:
                        captions = generate_captions(img, args.count)

                        captions_str = "  ".join(captions)

                        if args.words:
                            captions_str = " ".join(captions_to_words(captions))

                        batch_results.append((path, captions_str))
                
                results.extend(batch_results)

                if csv_writer:
                    csv_writer.writerows(batch_results)
                    csv_file.flush()
                    os.fsync(csv_file.fileno())

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

if csv_file:
    csv_file.close()

# Summary output
print("\n===== SUMMARY =====")
print(f"Total images found: {len(image_files)}")
print(f"Total images processed: {total_processed}")
print(f"Failed to load: {failed_loads}")

# Save to CSV if output file specified
if args.output:
    print(f"Results saved to: {args.output}")

# Print all results
print("\nResults:")
for path, caption in results:
    print(f"{path} → {caption}")
    if args.show:
        show(path, caption)

