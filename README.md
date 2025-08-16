# ai-image-desc
trying to describe images using ai

## Preview

### Images:
```
images/test1.jpg
images/test2.jpg
images/test3.jpeg
```

<table>
    <tr>
        <td><img src="images/test1.jpg" width="150" height="225"/></td>
        <td><img src="images/test2.jpg" width="200" height="225"/></td>
        <td><img src="images/test3.jpeg" width="200" height="225"/></td>
    </tr>
</table>

### Running
```sh
./describe.py -g images
```

### Gives
```
images/test1.jpg → a car driving down a dirt road in the fall
images/test2.jpg → a person standing on top of a sand dune
images/test3.jpeg → a boat is in the water near a city
```


## Setup & Install
### Create virtual environment
```sh
./setup-env.sh        # nvidia gpu
# or
./setup-env-rocm.sh   # amd gpu
```

### Install
```sh
./install.sh
# Creates symlinks to shell
# scripts that run the
# python scripts in a venv.
```

## aiid-describe --help
```console
usage: aiid-describe.py [-h] [-s] [-g] [-b BATCH] [-lb LOAD_BATCH] [-o OUTPUT] input

Describe images in English using BLIP

positional arguments:
  input                 image file or directory with images

options:
  -h, --help            show this help message and exit
  -s, --show            show image with label in title
  -g, --gpu             use gpu
  -b, --batch BATCH     batch size for GPU processing
  -lb, --load_batch LOAD_BATCH
                        batch size for loading images into memory
  -o, --output OUTPUT   output CSV file to save results
```

## aiid-nsfw-detector --help
(modded describe.py to detect nsfw words auto moderation for a web platform)
```console
usage: aiid-nsfw-detector.py [-h] [-s] [-sc] [-sn] [-g] [-b BATCH] [-lb LOAD_BATCH] input

Detect NSFW images using BLIP (auto moderation)

positional arguments:
  input                 image file or directory with images

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

## aiid-group --help
(needs manual tweaking to actually work well)
```console
usage: aiid-group.py [-h] [-s] [-g] directory

group images by eng names

positional arguments:
  directory   dir with images

options:
  -h, --help  show this help message and exit
  -s, --show  show image with label in title
  -g, --gpu   use gpu
```
