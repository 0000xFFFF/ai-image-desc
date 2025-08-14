# ai-image-desc
trying to describe images using ai

## describe.py
```
usage: describe.py [-h] [-s] [-g] [-b BATCH] [-lb LOAD_BATCH] [-o OUTPUT] directory

Describe images in English using BLIP

positional arguments:
  directory             dir with images

options:
  -h, --help            show this help message and exit
  -s, --show            show image with label in title
  -g, --gpu             use gpu
  -b, --batch BATCH     batch size for GPU processing
  -lb, --load_batch LOAD_BATCH
                        batch size for loading images into memory
  -o, --output OUTPUT   output CSV file to save results
```

## nsfw-detector.py
(modded describe.py to detect nsfw words auto moderation for a web platform)
```
usage: nsfw-detector.py [-h] [-s] [-sc] [-sn] [-g] [-b BATCH] [-lb LOAD_BATCH] directory

Detect NSFW images using BLIP (auto moderation)

positional arguments:
  directory             dir with images

options:
  -h, --help            show this help message and exit
  -s, --show            show images (clean+nsfw) with label in title
  -sc, --show_clean     show clean images with label in title
  -sn, --show_nsfw      show nsfw images with label in title
  -g, --gpu             use gpu
  -b, --batch BATCH     batch size for GPU processing
  -lb, --load_batch LOAD_BATCH
                        batch size for loading images into memory
```

## group.py
```
usage: group.py [-h] [-s] [-g] directory

group images by eng names

positional arguments:
  directory   dir with images

options:
  -h, --help  show this help message and exit
  -s, --show  show image with label in title
  -g, --gpu   use gpu
```
