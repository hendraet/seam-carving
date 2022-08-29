# Seam Carving for Text Line Extraction on Color and Grayscale Historical Manuscripts

Unofficial Python3 implementation for [this](https://infoscience.epfl.ch/record/198756/files/ICFHR_2014.pdf) paper.

## Installation

The code was tested using python 3.8.5.
To install all necessary dependencies run `pip install -r requirements.txt`.

## Usage

The separating seams can be calculated and drawn onto the original image by running: 
`python segment_lines.py [list of image names]`  
- The default output directory is `images`. It can be changed by supplying a different directory to the `--output-dir` flag.  
- Adding the `--debug` flag will also renderender calculated medial seams as well as the slices.
- The three hyperparameters `r`, `b` and `sigma` can be tuned using `--r`, `--b` and `--sigma` respectively.
