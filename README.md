# JPEG-Like Image Compression System

This repository contains the implementation of a modular, JPEG-inspired image compression and decompression system written in Python. Developed as part of the final project for EECE 5698: Visual Sensing and Computing at Northeastern University, the system is designed for educational exploration of lossy image compression algorithms, parameter sweep experimentation, and semantic evaluation via classification.

## Overview
Our system emulates the classical JPEG pipeline:
- RGB to YCbCr color space conversion
- Chroma downsampling (configurable)
- Block-based 2D Discrete Cosine Transform (DCT)
- Quantization with adjustable strategies
- Entropy encoding via custom Huffman coding
- Full decompression pipeline

In contrast to traditional implementations, this system provides:
- Flexible YAML-based configuration
- Pluggable quantization strategies (Gaussian, L-N Norm, etc.)
- Extensive benchmarking scripts
- Semantic preservation evaluation using pre-trained classifiers (e.g., ResNet50)

## Requirements & Installation

### Python Version
- Python 3.12+

### Required Packages
Install dependencies using pip:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is missing, install manually:
```bash
pip install numpy scipy scikit-image opencv-python Pillow matplotlib imageio rawpy pandas torch torchvision
```

> **Note:** AVIF and WebP image support is included via `pillow-avif-plugin`.

Or use GitHub Actions to retrieve the latest compiled version (see below).
## Repository Structure
```
├── README.md
├── requirements.txt
├── src/                             # Core compression & decompression source code
│   ├── compression.py
│   ├── decompression.py
│   ├── huffman.py
│   ├── utilities.py
│   └── main.py
├── test/                            # Test scripts and evaluation tools
│   ├── classification_tests.py
│   ├── parameter_sweeps.py
│   ├── results_compression.py
│   └── results_decompression.py
├── assets/                          # Test images and research references
│   ├── test_images/                # .CR2, .webp, .tif, etc. for compression input
│   └── unit_test_images/          # Small 16x16 synthetic images
├── compression_configurations/     # YAML config files for sweep experiments
│   ├── baseline_jpeg_sweep/
│   ├── block_size_sweep/
│   ├── downsample_sweep/
│   ├── quantization_sweep/
│   └── ...
├── results/                         # Saved .rde files and metrics.json from compression runs
│   ├── <timestamped_sweep_name>/
├── notebooks/
│   └── plot_results.ipynb          # Jupyter notebook for analyzing compression metrics
├── final-report/                   # LaTeX source and compiled PDF of final writeup
│   ├── main.tex
│   ├── main.pdf
│   ├── main.bib
│   ├── cvpr.sty
│   └── sec/
│       ├── 0_abstract.tex
│       ├── 1_intro.tex
│       └── ...
├── huffman_tables_comp.json        # Optional pre-generated Huffman tables
└── tmp/                             # Temporary output files from test runs
```

## How to Use

### 1. Compress an Image
Run the main compression pipeline:

```bash
python3 src/compression.py
```

This will:
- Load a test image and YAML config from the hardcoded block at the bottom
- Run the JPEG-like compression process
- Save the `.rde` binary and `.metrics.json` file

Modify this block in `compression.py` to change inputs:
```python
test_image_path = "assets/test_images/subject_cat.webp"
compression_config = "compression_configurations/quantization_sweep/gauss_quantization_most_agressive.yaml"
```

### 2. Decompress an Image
Reconstruct an `.rde` compressed image with:

```bash
python3 src/decompression.py
```

The script:
- Loads a hardcoded path to the `.rde` file
- Runs full decompression (Huffman decode, inverse DCT, etc.)
- Saves or displays the restored image

### 3. Run Parameter Sweeps
Benchmark multiple configurations across block sizes, downsampling levels, and quantization styles:

```bash
python3 test/parameter_sweeps.py
```

Results are saved under:
```
test/results/<sweep_name>/
```

Each sweep logs:
- Compression metrics
- Semantic classification results
- Runtime performance

## Visualizing Results
Run the Jupyter notebook to analyze trends:

```bash
jupyter notebook notebooks/plot_results.ipynb
```

Plots include:
- Compression ratio vs. quality
- Classification drift (ResNet50)
- Timing breakdowns
- PSNR/SSIM heatmaps

## Final-Report PDF
The compiled version of the final project report is automatically built via GitHub Actions:

- Visit the [Actions tab](../../actions)
- Select the latest successful `pdflatex` workflow
- Download `main.pdf` from the **compiled-pdf** artifact

> A formal release with the report will be made available under the [Releases tab](../../releases) soon.

## Key Features
- Modular JPEG-like compression pipeline
- Raw image (.CR2) and AVIF/WebP support
- Semantic classification accuracy testing
- Luminance vs chrominance quantization control
- Detailed benchmarking + config sweep framework
- GitHub Actions CI for LaTeX compilation

## Authors
Nicolas Drager, Muhammad Elarbi, George Roudebush,  
Department of Electrical and Computer Engineering  
Northeastern University  
`{drager.k, elarbi.m, roudebush.g,}@northeastern.edu`

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/).  
<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1" alt="">
<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1" alt="">
<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/nc.svg?ref=chooser-v1" alt="">
<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/sa.svg?ref=chooser-v1" alt="">

---
Developed for *EECE 5698: Special Topics: Visual Sensing and Computing,  Spring 2025*