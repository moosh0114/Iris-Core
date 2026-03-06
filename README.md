# Iris-Core

[![GitHub last commit](https://img.shields.io/github/last-commit/moosh0114/Iris-Core.svg)](https://github.com/moosh0114/Iris-Core)
[![GitHub repo size](https://img.shields.io/github/repo-size/moosh0114/Iris-Core.svg)](https://github.com/moosh0114/Iris-Core)

<br>
<!-- Future logo will be placed here -->

### Visual Weight Prediction Network for Color-Extraction

IMPORTANT : This project is still in the development and testing stages, licensing terms may be updated in the future. Please don't do any commercial usage currently.

## Project Dependencies Guide

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://github.com/pytorch/pytorch)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![scikit-learn](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://github.com/scikit-learn/scikit-learn)

**( ML & Data Pipeline )**

**[ for Dependencies Details please see the end of this README ]**

Iris-Core is the machine learning backbone of the Iris project. It uses PyTorch to build a Multi-Layer Perceptron (MLP) model that predicts a "Visual Importance Score" for extracted colors. Instead of relying solely on pixel count, it evaluates OKLCH color space values and spatial centrality to rank color palettes based on human visual perception. 

Iris-Core uses uv for dependency and environment management. uv has multiple licenses. PyTorch is licensed under the BSD-style license. NumPy is licensed under the BSD 3-Clause License.

## Quickstart ( CLI )

**Build Dependencies ( Install uv )**

upgrade : `python -m pip install --upgrade pip`

use uv : `python -m pip install uv` & `uv sync`

**Run the Inference Pipeline**

You can run the main script to test the model architecture with dummy data:

```shell
uv run python main.py
```

**Run the Model Architecture Test**

To verify the PyTorch model's tensor dimensions and output logic:

```shell
uv run python models/model.py
```

## Project Dependencies Details

PyTorch License : https://github.com/pytorch/pytorch/blob/main/LICENSE

NumPy License : https://github.com/numpy/numpy/blob/main/LICENSE.txt

scikit-learn License : https://github.com/scikit-learn/scikit-learn?tab=BSD-3-Clause-1-ov-file#readme

uv License : https://github.com/astral-sh/uv/blob/main/LICENSE-MIT & another Apache-2.0 License