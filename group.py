#!/usr/bin/env python

import numpy as np
import torch
import clip
from sklearn.cluster import KMeans
from PIL import Image
import os
from sklearn.metrics.pairwise import cosine_similarity
import glob
import argparse

# Argument parsing
parser = argparse.ArgumentParser(description='group images by eng names')
parser.add_argument('directory', type=str, help="dir with images")
parser.add_argument('-d', '--display', action='store_true', help="show image with label in title")
parser.add_argument('-g', '--gpu', action='store_true', help="use gpu")
args = parser.parse_args()

print(f"searching in dir: {args.directory}")

if args.display:
    import matplotlib.pyplot as plt
    from skimage import io

device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load CLIP
model, preprocess = clip.load("ViT-B/32", device=device)

# Find images recursively
image_files = []
for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
    image_files.extend(glob.glob(os.path.join(args.directory, "**", ext), recursive=True))

# Load and embed images
embeddings = []
for filepath in image_files:
    try:
        img = preprocess(Image.open(filepath)).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model.encode_image(img).cpu().numpy()
        embeddings.append(emb[0])
    except Exception as e:
        print(f"Skipping {filepath}: {e}")

embeddings = np.array(embeddings)

# Cluster
n_clusters = min(5, len(embeddings))
kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(embeddings)
labels = kmeans.labels_

# Average embedding per cluster
cluster_embs = []
for cluster_id in range(n_clusters):
    cluster_embs.append(embeddings[labels == cluster_id].mean(axis=0))
cluster_embs = np.array(cluster_embs)

# Candidate labels
candidate_labels = ["man", "woman", "people", "dog", "cat", "beach", "city", "forest", "food", "car", "mountains", "flowers"]

# Encode text prompts
with torch.no_grad():
    text_tokens = clip.tokenize(candidate_labels).to(device)
    text_embs = model.encode_text(text_tokens).cpu().numpy()

# Match clusters to labels
for i, cluster_vector in enumerate(cluster_embs):
    sims = cosine_similarity([cluster_vector], text_embs)[0]
    best_idx = np.argmax(sims)
    print(f"Cluster {i} → {candidate_labels[best_idx]}")

# Show image assignments
for fname, cluster_id in zip(image_files, labels):
    label_name = candidate_labels[np.argmax(cosine_similarity([cluster_embs[cluster_id]], text_embs)[0])]
    print(f"{fname} → {label_name}")

    if args.display:
        plt.figure()
        plt.title(label_name)
        plt.imshow(io.imread(fname))
        plt.axis("off")
        plt.show()
