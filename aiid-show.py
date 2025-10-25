#!/usr/bin/env python
import argparse
import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import re

def read_csv(file):
    results = []
    try:
        with open(file, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile, delimiter='|')
            headers = next(reader)  # Skip header
            for row in reader:
                results.append(row)
    except Exception as e:
        print(f"Failed to read CSV: {e}")
    return results

def show_images(results):
    for row in results:
        print(row)
        path = row[0]
        caption = row[1]
        title = row[2] if len(row) > 2 else ""

        if os.path.exists(path):
            img = mpimg.imread(path)
            plt.imshow(img)
            plt.axis('off')
            plt.title(caption)
            plt.suptitle(title, fontsize=10)
            plt.show()
        else:
            print(f"File not found: {path}")

def main():
    parser = argparse.ArgumentParser(description="Display images from CSV with captions.")
    parser.add_argument('csv_file', metavar="file.csv", help="Path to the CSV file")
    parser.add_argument('-n', '--nsfw', action='store_true', help="filter out NSFW images by matching words in nsfw.lst")
    args = parser.parse_args()

    results = read_csv(args.csv_file)
    found = []
    
    if args.nsfw:
        nsfw_keywords = set()
        nsfw_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "nsfw.lst")
        with open(nsfw_file, "r") as f:
            for line in f:
                x = line.strip()
                if x not in nsfw_keywords:
                    nsfw_keywords.add(x)

        print(f"Loaded {len(nsfw_keywords)} nsfw keywords")

        # Regex pattern for NSFW check
        pattern = re.compile(r"\\b(" + "|".join(map(re.escape, nsfw_keywords)) + r")\\b", re.IGNORECASE)

        for result in results:
            pattern = re.compile(r"\b(" + "|".join(map(re.escape, nsfw_keywords)) + r")\b", re.IGNORECASE)
            file_path = result[0]
            text = result[1]
            match = pattern.search(text)
            if match:
                found.append((file_path, text, match.group(0)))
    
        show_images(found)
    
    else:
        show_images(results)

if __name__ == "__main__":
    main()

