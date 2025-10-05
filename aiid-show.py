#!/usr/bin/env python
import argparse
import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def read_csv(file):
    results = []
    try:
        with open(file, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
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
        if os.path.exists(path):
            img = mpimg.imread(path)
            plt.imshow(img)
            plt.axis('off')
            plt.title(caption)
            plt.show()
        else:
            print(f"File not found: {path}")

def main():
    parser = argparse.ArgumentParser(description="Display images from CSV with captions.")
    parser.add_argument('csv_file', metavar="file.csv", help="Path to the CSV file")
    args = parser.parse_args()

    results = read_csv(args.csv_file)
    show_images(results)

if __name__ == "__main__":
    main()

