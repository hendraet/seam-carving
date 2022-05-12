# Seam Carving for Text Line Extraction on Color and Grayscale Historical Manuscripts

Unofficial Python3 implementation for [this](https://os.zhdk.cloud.switch.ch/tind-tmp-epfl/a8cbb0eb-9124-475e-8915-528feb181a1c?response-content-disposition=attachment%3B%20filename%2A%3DUTF-8%27%27ICFHR_2014.pdf&response-content-type=application%2Fpdf&AWSAccessKeyId=ded3589a13b4450889b2f728d54861a6&Expires=1652434717&Signature=s4ZofQs3n64W8LXAC1%2BvTcmH2%2BE%3D) paper.

## Installation

The code was tested using python 3.8.5.
To install all necessary dependencies run `pip install -r requirements.txt`.

## Usage

The separating seams can be calculated and drawn onto the original image by running: 
`python segment_lines.py [list of image names]`  
- The default output directory is `images`. It can be changed by supplying a different directory to the `--output-dir` flag.  
- Adding the `--debug` flag will also renderender calculated medial seams as well as the slices.
- The three hyperparameters `r`, `b` and `sigma` can be tuned using `--r`, `--b` and `--sigma` respectively.
