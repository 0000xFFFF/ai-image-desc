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
parser.add_argument('-s', '--show', action='store_true', help="show image with label in title")
parser.add_argument('-g', '--gpu', action='store_true', help="use gpu")
parser.add_argument('-b', '--batch', type=int, default=8, help="batch size for GPU processing")
parser.add_argument('-lb', '--load_batch', type=int, default=256, help="batch size for loading images into memory")
parser.add_argument('-o', '--output', type=str, help="output CSV file to save results")
args = parser.parse_args()

if args.show:
    import matplotlib.pyplot as plt
    from skimage import io

def show(image_path, caption):
    plt.figure()
    plt.title(caption)
    plt.imshow(io.imread(image_path))
    plt.axis("off")
    plt.show()

print(f"Searching in: {args.input}")

# Collect image files
image_files = []
if os.path.isfile(args.input):
    # Single file
    image_files = [args.input]
elif os.path.isdir(args.input):
    # Directory (recursive search)
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        image_files.extend(glob.glob(os.path.join(args.input, "**", ext), recursive=True))
else:
    print("Error: Input path is neither a file nor a directory")
    exit()

if not image_files:
    print("No images found!")
    exit()

print(f"Found {len(image_files)} images")

# Device
device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load BLIP
from transformers import BlipProcessor, BlipForConditionalGeneration
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base",
    use_fast=False
)
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
model.to(device)
model.eval()

if not image_files:
    print("No images found!")
    exit()

print(f"Found {len(image_files)} images")
print(f"Processing batch size: {args.batch}")
print(f"Loading batch size: {args.load_batch}")

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

# Process images in loading batches to manage memory
load_batch_size = args.load_batch
processing_batch_size = args.batch

# Create nested progress bars
with tqdm(total=len(image_files), desc="Overall progress", position=0) as overall_pbar:
    with tqdm(total=0, desc="Current batch", position=1, leave=False) as batch_pbar:
        
        for load_start in range(0, len(image_files), load_batch_size):
            # Get current batch of file paths
            load_end = min(load_start + load_batch_size, len(image_files))
            current_batch_paths = image_files[load_start:load_end]
            
            # Reset and configure batch progress bar
            batch_pbar.reset(total=len(current_batch_paths))
            batch_pbar.set_description(f"Loading batch {load_start//load_batch_size + 1}/{(len(image_files)-1)//load_batch_size + 1}")
            
            # Load current batch of images using thread pool
            loaded_images = []
            with ThreadPoolExecutor(max_workers=8) as executor:
                for result in executor.map(load_image, current_batch_paths):
                    loaded_images.append(result)
                    batch_pbar.update(1)
            
            # Filter out failed loads
            valid_images = [(p, img) for p, img in loaded_images if img is not None]
            failed_loads += len(loaded_images) - len(valid_images)
            
            if not valid_images:
                overall_pbar.update(len(current_batch_paths))
                continue
            
            # Reset batch progress bar for processing
            batch_pbar.reset(total=len(valid_images))
            batch_pbar.set_description(f"Processing batch {load_start//load_batch_size + 1}")
            
            # Process the loaded images in smaller processing batches
            for proc_start in range(0, len(valid_images), processing_batch_size):
                proc_end = min(proc_start + processing_batch_size, len(valid_images))
                batch = valid_images[proc_start:proc_end]
                paths, images = zip(*batch)
                
                # Process batch with BLIP
                inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
                
                with torch.no_grad():
                    with torch.amp.autocast("cuda", enabled=(device=="cuda")):
                        outputs = model.generate(**inputs)
                
                captions = [processor.decode(o, skip_special_tokens=True) for o in outputs]
                
                # Store results
                for path, caption in zip(paths, captions):
                    results.append((path, caption))
                    if args.show:
                        show(path, caption)
                
                total_processed += len(batch)
                batch_pbar.update(len(batch))
                overall_pbar.update(len(batch))
                
                # Clear GPU cache if using CUDA
                if device == "cuda":
                    torch.cuda.empty_cache()
            
            # Clear the loaded images from memory
            del valid_images
            del loaded_images
            gc.collect()  # Force garbage collection
            
            # Clear GPU cache again
            if device == "cuda":
                torch.cuda.empty_cache()

# Summary output
print("\n===== SUMMARY =====")
print(f"Total images found: {len(image_files)}")
print(f"Total images processed: {total_processed}")
print(f"Failed to load: {failed_loads}")

# Save to CSV if output file specified
if args.output:
    try:
        with open(args.output, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Path', 'Caption'])  # Header
            writer.writerows(results)
        print(f"Results saved to: {args.output}")
    except Exception as e:
        print(f"Failed to save CSV: {e}")

# Print all results
print("\nResults:")
for path, caption in results:
    print(f"{path} → {caption}")
