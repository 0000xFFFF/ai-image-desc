#!/usr/bin/env python
import torch
from PIL import Image
import os
import glob
import argparse
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import gc

# Argument parsing
parser = argparse.ArgumentParser(description='Detect NSFW images using BLIP (auto moderation)')
parser.add_argument('directory', type=str, help="dir with images")
parser.add_argument('-s', '--show', action='store_true', help="show images (clean+nsfw) with label in title")
parser.add_argument('-sc', '--show_clean', action='store_true', help="show clean images with label in title")
parser.add_argument('-sn', '--show_nsfw', action='store_true', help="show nsfw images with label in title")
parser.add_argument('-g', '--gpu', action='store_true', help="use gpu")
parser.add_argument('-b', '--batch', type=int, default=8, help="batch size for GPU processing")
parser.add_argument('-lb', '--load_batch', type=int, default=256, help="batch size for loading images into memory")
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

print(f"Found {len(image_files)} images")
print(f"Processing batch size: {args.batch}")
print(f"Loading batch size: {args.load_batch}")

# NSFW keywords
nsfw_keywords = [
    "nude", "naked", "sexual", "porn", "breast", "penis", "vagina",
    "buttocks", "sex", "erotic", "explicit", "intimate"
]

# Threaded image loading function
def load_image(path):
    try:
        img = Image.open(path).convert("RGB")
        return path, img
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        return path, None

# Results storage
nsfw_results = []
clean_results = []
total_processed = 0
failed_loads = 0

# Process images in loading batches to manage memory
load_batch_size = args.load_batch
processing_batch_size = args.batch

# Create progress bar for overall progress
overall_progress = tqdm(total=len(image_files), desc="Overall progress")

for load_start in range(0, len(image_files), load_batch_size):
    # Get current batch of file paths
    load_end = min(load_start + load_batch_size, len(image_files))
    current_batch_paths = image_files[load_start:load_end]
    
    print(f"\nLoading batch {load_start//load_batch_size + 1}/{(len(image_files)-1)//load_batch_size + 1}")
    print(f"Loading images {load_start+1} to {load_end}")
    
    # Load current batch of images using thread pool
    with ThreadPoolExecutor(max_workers=8) as executor:
        loaded_images = list(tqdm(
            executor.map(load_image, current_batch_paths), 
            total=len(current_batch_paths), 
            desc="Loading images",
            leave=False
        ))
    
    # Filter out failed loads
    valid_images = [(p, img) for p, img in loaded_images if img is not None]
    failed_loads += len(loaded_images) - len(valid_images)
    
    if not valid_images:
        overall_progress.update(len(current_batch_paths))
        continue
    
    print(f"\nSuccessfully loaded {len(valid_images)} images from this batch\n")
    
    # Process the loaded images in smaller processing batches
    for proc_start in range(0, len(valid_images), processing_batch_size):
        proc_end = min(proc_start + processing_batch_size, len(valid_images))
        batch = valid_images[proc_start:proc_end]
        paths, images = zip(*batch)
        
        # Process batch with BLIP
        inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(device=="cuda")):
                outputs = model.generate(**inputs)
        
        captions = [processor.decode(o, skip_special_tokens=True) for o in outputs]
        
        # Analyze results
        for path, caption in zip(paths, captions):
            if any(word in caption.lower() for word in nsfw_keywords):
                nsfw_results.append((path, caption))
            else:
                clean_results.append((path, caption))
        
        total_processed += len(batch)
        overall_progress.update(len(batch))
        
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

overall_progress.close()

# Summary output
print("\n===== SUMMARY =====")
print(f"Total images found: {len(image_files)}")
print(f"Total images processed: {total_processed}")
print(f"Failed to load: {failed_loads}")
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
