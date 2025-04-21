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
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── src/                            # Core compression/decompression logic
│   ├── compression.py              # FlexibleJPEG class: compression pipeline
│   ├── decompression.py            # FlexibleJPEGDecompress class: decompression pipeline
│   ├── huffman.py                  # Custom Huffman tree + encoder/decoder
│   ├── utilities.py                # Zigzag scan, quantization matrix generators, padding
│   └── main.py                     # Optional unified CLI entry point (can be expanded)
├── test/                           # Experimental and evaluation scripts
│   ├── classification_tests.py     # Top-1 drift testing with ResNet50
│   ├── parameter_sweeps.py         # Batch sweep runner (block size, quant, etc.)
│   ├── results_compression.py      # Outputs metrics + rde for single image
│   └── results_decompression.py    # Reconstructs + logs from `.rde` output
├── compression_configurations/     # YAML configurations for different sweep strategies
│   ├── baseline_jpeg_sweep/        # JPEG-matching quantization quality levels
│   ├── block_size_sweep/           # Block size variations (e.g., 8x8, 16x16, etc.)
│   ├── downsample_sweep/           # Chroma subsampling schemes (e.g., 4:4:4, 4:2:0)
│   ├── quantization_sweep/         # Gaussian-inspired quantization tables
│   ├── quantization_sweep_chroma/  # Targeted quantization for chroma channels
│   ├── quantization_sweep_luma/    # Targeted quantization for luma channel
│   └── quantization_sweep_small_blocks/ # Small block + high quantization stress tests
├── assets/
│   ├── test_images/                # Real-world and synthetic input images (.cr2, .webp, .tif, .png)
│   └── unit_test_images/          # 16×16 synthetic grids for debugging
├── results/                        # `.rde` binaries and `.metrics.json` logs
│   └── <timestamped_run>/         # Output from `parameter_sweeps.py` or other tools
├── notebooks/
│   └── plot_results.ipynb         # Jupyter notebook: visualizations for compression trends
├── final-report/                   # CVPR-style final paper for course submission
│   ├── main.tex                    # Entry point LaTeX file
│   ├── main.pdf                    # Compiled report (also available via GitHub Actions)
│   ├── main.bib                    # BibTeX source
│   ├── cvpr.sty                    # Conference style file
│   └── sec/                        # Individual sections of paper in modular .tex files
├── huffman_tables_comp.json        # Pre-trained symbol frequency table for encoding
├── tmp/                            # Temp folder used for on-the-fly compression results
└── venv/                           # Python virtual environment (excluded from version control)
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