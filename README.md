# ai-image-desc

describe image in English with AI

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
./aiid-describe -g images
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
./setup-venv.sh        # nvidia gpu
# or
./setup-venv-rocm.sh   # amd gpu
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
usage: aiid-describe.py [-h] [-s] [-c number] [-w] [-wc] [-g] [-b number]
                        [-lb number] [-o file.csv] [-od file.csv] [-d] [-d2]
                        input

Describe images in English using BLIP

positional arguments:
  input                 image file or directory with images

options:
  -h, --help            show this help message and exit
  -s, --show            show image(s) with caption in title after processing
                        all
  -c, --count number    how many captions to generate per image (default: 1)
  -w, --words           turn captions into keywords that will be sorted
  -wc, --words_clean    same as -w + remove function words like ('the', 'a',
                        'an', ...)
  -g, --gpu             use gpu
  -b, --batch number    batch size for GPU processing
  -lb, --load_batch number
                        batch size for loading images into memory
  -o, --output file.csv
                        output results to CSV file
  -od, --output_delimiter file.csv
                        delimiter to use when outputing csv file (default:
                        '|')
  -d, --defaults        -c 5 -wc -g -b 128 -lb 256 -o output.csv
  -d2, --defaults2      -c 5 -wc -g -b 256 -lb 512 -o output.csv
```

### Recommended way to run

```sh
./aiid-describe -g images -d
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
