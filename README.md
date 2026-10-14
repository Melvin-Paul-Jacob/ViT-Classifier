# Vision Transformer from Scratch with PyTorch

A from-scratch implementation of a **Vision Transformer (ViT)** for image classification using PyTorch. The model is trained and evaluated on **Imagenette**, a 10-class subset of ImageNet.

<p align="center">
  <img src="src/docs/images/sample_images.jpg" alt="Imagenette sample images" width="850">
</p>

## Table of Contents

- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Dataset and Preprocessing](#dataset-and-preprocessing)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Training](#training)
- [Evaluation and Predictions](#evaluation-and-predictions)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Acknowledgements](#acknowledgements)

---

## Overview

Vision Transformers apply the Transformer architecture to images by dividing them into patches and processing them as a sequence of tokens.

This project implements the core ViT components from scratch:

- Patch embeddings using `Conv2d`
- Learnable classification and positional embeddings
- Multi-head self-attention
- Transformer encoder blocks with residual connections
- MLP classification head
- Training and validation pipelines
- Data augmentation and learning-rate scheduling

### Model workflow

```mermaid
flowchart LR
    A["Input Image"] --> B["Patch Embedding"]
    B --> C["Add CLS Token and Positions"]
    C --> D["Transformer Encoder"]
    D --> E["Extract CLS Token"]
    E --> F["Classification Head"]
    F --> G["Class Predictions"]
```

## Model Architecture

The model follows a pre-normalization Transformer design. Each encoder block contains multi-head self-attention, an MLP, layer normalization, and residual connections.

For an input image of size 224 × 224 with a patch size of 16 × 16:

- Number of image patches: 196
- Number of tokens, including the CLS token: 197
- Output: scores for 10 Imagenette classes

The classification token's final representation is passed to a linear layer to predict the image category.

## Dataset and Preprocessing

The project uses [Imagenette](https://github.com/fastai/imagenette) through `torchvision.datasets.Imagenette`.

The dataset contains 10 ImageNet categories, including tench, English springer, cassette player, church, golf ball, and parachute.

**Training preprocessing**
- Random resized crop
- Random horizontal flip
- Color jitter
- Random rotation
- ImageNet normalization

**Validation preprocessing**
- Image resizing
- Conversion to tensor
- ImageNet normalization

Training augmentations help improve generalization, while deterministic validation preprocessing supports consistent evaluation.

## Project Structure

```text
Vision_Transformer/
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── test.py
│   └── docs/
│       └── images/
├── outputs/
│   ├── checkpoints/
│   │   └── best_model.pt
│   └── history/
│       └── train_history.json
├── requirements.txt
└── README.md
```

| File | Purpose |
|---|---|
| `config.py` | Model and training configuration |
| `dataset.py` | Dataset loading and preprocessing |
| `model.py` | Vision Transformer implementation |
| `train.py` | Training and validation |
| `test.py` | Inference and prediction visualization |
| `outputs/` | Saved checkpoints and training history |

## Installation

**Requirements:** Python 3.10+, PyTorch, Torchvision, NumPy, Matplotlib, Pillow, and tqdm. A CUDA-compatible GPU is recommended but not required.

Clone the repository and install its dependencies:

```bash
git clone https://github.com/your-user/vision-transformer-pytorch.git
cd vision-transformer-pytorch

python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install the appropriate PyTorch build for your hardware using the [official installation guide](https://pytorch.org/get-started/locally/).

## Configuration

The model and training settings are managed through `vit_config` in `config.py`.

| Parameter | Example value |
|---|---:|
| Image size | 224 |
| Patch size | 16 |
| Embedding dimension | 384 |
| Attention heads | 6 |
| Encoder layers | 6 |
| Number of classes | 10 |
| Batch size | 32 |
| Training epochs | 30 |
| Learning rate | 0.0003 |

These are example settings; your actual configuration may differ.

## Training

Run the training script from the project root:

```bash
python src/train.py
```

The training pipeline loads Imagenette, initializes the model, optimizes the classification loss, evaluates validation performance after each epoch, and uses cosine learning-rate scheduling.

The best checkpoint is saved at:

```text
outputs/checkpoints/best_model.pt
```

Training history is stored in `outputs/history/train_history.json`.

## Evaluation and Predictions

Run inference on randomly selected validation images:

```bash
python src/test.py
```

The script loads the trained checkpoint and displays sample validation images alongside:

- Ground-truth labels
- Predicted classes
- Prediction confidence
- Top-5 class probabilities

<p align="center">
  <img src="src/docs/images/predictions.png" alt="ViT prediction visualization" width="850">
</p>

## Results

The current checkpoint achieved a **validation accuracy of 73.94%** after 30 epochs.

| Metric | Result |
|---|---:|
| Training epochs | 30 |
| Number of classes | 10 |
| Validation accuracy | 73.94% |

The reported accuracy is from the saved checkpoint. Results may vary depending on model configuration, random initialization, augmentation, and training settings.

## Future Improvements

Possible extensions include:

- Mixed-precision training
- AdamW optimization and learning-rate warmup
- Mixup and CutMix augmentation
- Confusion matrix and per-class metrics
- Attention-map visualization
- TensorBoard or Weights & Biases logging
- Pretraining and fine-tuning on larger datasets

## Acknowledgements

- [An Image Is Worth 16×16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [PyTorch](https://pytorch.org/)
- [Torchvision](https://pytorch.org/vision/stable/index.html)
- [Imagenette](https://github.com/fastai/imagenette)

## License

MIT License

Copyright (c) 2026 Melvin Paul Jacob

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Contributing

Contributions are welcome! If you would like to improve this project, fix a bug, optimize the model, or add new features, feel free to contribute.

1. Fork this repository.
2. Create a new branch for your changes.
3. Make your changes and test them.
4. Submit a pull request with a clear description of your improvements.

Please keep contributions clean, well-documented, and focused on improving the project.

By submitting a contribution, you agree that it may be distributed under the project's MIT License.

If you find a bug or have an idea for improvement, feel free to open an issue.

**Let's learn, build, and improve together!**
