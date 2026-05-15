# SRGAN for OCR Enhancement on Handwritten Text

> ⚠️ **This project is currently under active development. Features, results, and documentation are subject to change.**

A deep learning pipeline that applies **Super-Resolution Generative Adversarial Networks (SRGAN)** to enhance low-resolution handwritten text images, with the goal of improving downstream OCR accuracy. The model is trained on the **IAM Handwriting Dataset** and upscales images by a factor of **4×**.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Loss Functions](#loss-functions)
- [Training Details](#training-details)
- [Project Structure](#project-structure)
- [Current Progress](#current-progress)
- [Planned Improvements](#planned-improvements)
- [Requirements](#requirements)
- [Getting Started](#getting-started)

---

## Overview

Handwritten text images captured in real-world conditions are often low-resolution, noisy, or degraded — making OCR pipelines less reliable. This project investigates whether super-resolution reconstruction using GANs can serve as an effective preprocessing step to improve text clarity before OCR.

The pipeline takes a low-resolution (LR) handwritten line image and reconstructs a visually sharper, high-resolution (HR) version using a generator network trained adversarially against a discriminator, with perceptual losses guided by VGG19 feature maps.

---

## Dataset

**IAM Handwriting Dataset** — a widely used benchmark for handwriting recognition research, containing segmented handwritten English text line images written by multiple authors.

**Image Resolutions used in this project:**

| Type | Height | Width |
|------|--------|-------|
| High-Resolution (HR) | 64 px | 512 px |
| Low-Resolution (LR) | 16 px | 128 px |

LR images are derived by bicubic downsampling of the HR images at a 4× scale factor. Images are normalized to the `[-1, 1]` range for training.

---

## Model Architecture

### Generator

The generator follows the standard SRGAN architecture adapted for text line images:

- Initial 9×9 convolution + PReLU activation
- 5 Residual Blocks, each containing:
  - 3×3 Conv → Batch Normalization → PReLU → 3×3 Conv → Batch Normalization → Skip Connection
- Post-residual 3×3 convolution + Batch Normalization
- Global skip connection from pre-residual features
- 2× Upscaling Blocks using PixelShuffle (`depth_to_space`) with 256 filters each
- Final 9×9 convolution with `tanh` activation for output reconstruction

**Input:** `(16, 128, 3)` — **Output:** `(64, 512, 3)`

### Discriminator

A CNN-based binary classifier that distinguishes real HR images from generated SR images:

- Initial 3×3 Conv + LeakyReLU (α = 0.2)
- 7 discriminator blocks progressively doubling filters (64 → 512) with alternating stride-1 and stride-2 convolutions, each with Batch Normalization + LeakyReLU
- Flatten → Dense(1024) → LeakyReLU → Dense(1, sigmoid)

---

## Loss Functions

Training uses a composite generator loss combining two components:

**Perceptual (VGG) Loss** — Mean squared error between VGG19 feature maps (layer 20, `block5_conv4`) of the real and generated HR images. This encourages perceptually meaningful reconstruction beyond pixel-level accuracy.

**Adversarial Loss** — Binary cross-entropy loss from the discriminator's response to generated images, weighted at `1e-3`:

```
Generator Loss = VGG Loss + 1e-3 × Adversarial Loss
```

The discriminator is trained with standard real/fake binary cross-entropy.

---

## Training Details

| Parameter | Value |
|-----------|-------|
| Framework | TensorFlow / Keras |
| Optimizer | Adam (lr = 5e-5, β₁ = 0.9) |
| Batch Size | 16 |
| Planned Epochs | 50 |
| Epochs Completed | 27 |
| Upscaling Factor | 4× |
| HR Resolution | 64 × 512 |
| LR Resolution | 16 × 128 |
| Residual Blocks | 5 |

Model checkpoints are saved every 10 epochs as `srgan_gen_epoch_<N>.h5`. Training progress is visualized by comparing LR input, SRGAN prediction, and HR ground truth side-by-side.

---

## Project Structure

```
.
├── srgan_ocr_enhancement.ipynb   # Main training notebook
├── srgan_gen_epoch_10.h5         # Generator checkpoint (epoch 10)
├── srgan_gen_epoch_20.h5         # Generator checkpoint (epoch 20)
└── README.md
```

---

## Current Progress

- ✅ Dataset extraction and preprocessing pipeline
- ✅ TF Data pipeline with batching, shuffling, and prefetching
- ✅ Generator architecture (residual blocks + PixelShuffle upsampling)
- ✅ Discriminator architecture (multi-scale CNN)
- ✅ VGG19-based perceptual loss
- ✅ Custom GradientTape training loop
- ✅ Training visualization per epoch
- ✅ Initial training completed (27/50 epochs)
- ✅ Sample super-resolution outputs generated
- 🔄 Training ongoing — full 50-epoch run in progress

---

## Planned Improvements

- Complete full 50-epoch training run
- Quantitative evaluation using **PSNR** and **SSIM** metrics
- **OCR benchmarking** with Tesseract OCR on SR vs. bicubic-upscaled outputs
- Experiment with **ESRGAN** (Enhanced SRGAN with Residual-in-Residual Dense Blocks)
- Incorporate **edge-aware enhancement** techniques suited for text
- Hyperparameter tuning (learning rate scheduling, more residual blocks)
- Evaluation on degraded/noisy real-world handwriting samples

---

## Requirements

```
tensorflow
numpy
opencv-python
pillow
matplotlib
scikit-learn
```

Install with:

```bash
pip install tensorflow numpy opencv-python pillow matplotlib scikit-learn
```

---

## Getting Started

1. Download the **IAM Handwriting Dataset** (lines split) and place `lines.tgz` in `/content/`.
2. Open `srgan_ocr_enhancement.ipynb` in Google Colab or a local Jupyter environment.
3. Run all cells sequentially — the notebook handles extraction, preprocessing, model building, and training.
4. Trained generator checkpoints are saved every 10 epochs.

---

## Technologies Used

Python · TensorFlow · Keras · NumPy · OpenCV · Pillow · Matplotlib · Scikit-learn · VGG19 (ImageNet pretrained)
