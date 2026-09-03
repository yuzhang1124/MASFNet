# Towards Accurate Image Segmentation with Multi-Scale Adaptive Feature  Modeling and Semantic Fusion
![Image](https://github.com/user-attachments/assets/f4f23a2d-532d-43e3-9849-525f60f72c62)

## Download The pretrained weights, prediction maps, and related experimental files can be downloaded from Baidu Netdisk: - Baidu Netdisk: [https://pan.baidu.com/s/1gnPu6M8ZLmKG3PeGQuCczA?pwd=htf5] - Extraction code: `htf5` The repository will be continuously updated with training scripts, testing scripts, model weights, and evaluation instructions.
# Towards Accurate Image Segmentation with Multi-Scale Adaptive Feature Modeling and Semantic Fusion

Official PyTorch implementation of **MASFNet**, corresponding to the manuscript:

**Towards Accurate Image Segmentation with Multi-Scale Adaptive Feature Modeling and Semantic Fusion**

<p align="center">
  <img src="https://github.com/user-attachments/assets/f4f23a2d-532d-43e3-9849-525f60f72c62" width="900">
</p>

## Overview

MASFNet is a unified binary image segmentation framework built upon the SAM2/Hiera-L backbone.

The proposed framework contains three main components:

- **MSPE (Multi-Scale Perception Enhancement):** enhances multi-scale feature representation using parallel convolutional branches with different receptive fields.
- **DCFF (Dual-Stream Cross Feature Fusion):** performs cross-level feature interaction through complementary channel and spatial attention.
- **CSGF (Cross-Level Semantic Guidance Fusion):** progressively integrates high-level semantic information and low-level spatial details during decoding.

The current repository provides the core implementation of MASFNet together with the training, testing, and evaluation pipeline used in our experiments.

---

## Repository Structure

```text
MASFNet/
│
├── MASFNet.py              # Main MASFNet architecture
├── dataset.py              # Training and testing dataset loader
├── train.py                # Training script
├── test.py                 # Inference script
├── eval.py                 # Quantitative evaluation script
│
├── train.sh                # Example training command
├── test.sh                 # Example inference command
├── eval.sh                 # Example evaluation command
│
├── requirements.txt        # Python dependencies
│
├── sam2/                   # SAM2 implementation
├── sam2_configs/           # SAM2/Hiera configuration files
│
├── LICENSE
└── README.md
```

---

## Environment

The experiments were implemented with PyTorch and conducted on a single NVIDIA RTX 3080 GPU.

The main training configuration used for SOD/COD experiments is:

- Input size: `352 × 352`
- Batch size: `6`
- Learning rate: `1e-4`
- Optimizer: `AdamW`
- Weight decay: `5e-4`
- Training epochs: `50`

We recommend using a CUDA-enabled environment.

### Installation

Clone the repository:

```bash
git clone https://github.com/yuzhang1124/MASFNet.git
cd MASFNet
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Additional packages used by the current implementation can be installed with:

```bash
pip install einops matplotlib opencv-python
```

The main dependencies include:

```text
Python
PyTorch
torchvision
torchaudio
hydra-core
einops
numpy
Pillow
imageio
opencv-python
pysodmetrics
```

---

## Pretrained SAM2/Hiera-L Backbone

MASFNet uses the pretrained **SAM2 Hiera-L** model as the backbone.

Please download the corresponding SAM2 Hiera-L pretrained checkpoint and specify its path using:

```bash
--hiera_path /path/to/sam2_hiera_large.pt
```

The corresponding SAM2/Hiera configuration files are included in:

```text
sam2_configs/
```

---

## Dataset Preparation

The training and testing images and ground-truth masks should be organized separately.

An example directory structure is:

```text
dataset/
├── Train/
│   ├── Image/
│   │   ├── xxx.jpg
│   │   ├── xxx.jpg
│   │   └── ...
│   │
│   └── GT/
│       ├── xxx.png
│       ├── xxx.png
│       └── ...
│
└── Test/
    ├── Image/
    │   ├── xxx.jpg
    │   ├── xxx.jpg
    │   └── ...
    │
    └── GT/
        ├── xxx.png
        ├── xxx.png
        └── ...
```

The image and ground-truth files should use corresponding filenames.

> **Note:** In the current dataset loader, directory paths are concatenated directly with filenames. Therefore, please ensure that the image and mask directory paths end with `/`.

For example:

```text
/path/to/DUTS-TR/Image/
/path/to/DUTS-TR/GT/
```

---

## Training

The default training protocol used in our SOD/COD experiments is:

```text
Input size   : 352 × 352
Epochs       : 50
Batch size   : 6
Learning rate: 1e-4
Optimizer    : AdamW
Weight decay : 5e-4
```

Run:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --hiera_path "/path/to/sam2_hiera_large.pt" \
    --train_image_path "/path/to/train/images/" \
    --train_mask_path "/path/to/train/masks/" \
    --save_path "./checkpoints/" \
    --epoch 50 \
    --lr 0.0001 \
    --batch_size 6
```

Alternatively, modify the paths in:

```bash
train.sh
```

and run:

```bash
bash train.sh
```

The trained checkpoints will be saved in the directory specified by:

```text
--save_path
```

---

## Testing

After training, predicted segmentation maps can be generated using `test.py`.

Example:

```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
    --checkpoint "./checkpoints/MASFNet-50.pth" \
    --test_image_path "/path/to/test/images/" \
    --test_gt_path "/path/to/test/masks/" \
    --save_path "./predictions/"
```

Alternatively:

```bash
bash test.sh
```

The predicted masks will be stored in:

```text
./predictions/
```

or the directory specified by `--save_path`.

---

## Evaluation

Quantitative evaluation is performed using `eval.py`.

The current evaluation code supports multiple commonly used binary segmentation metrics, including:

- MAE
- S-measure (`S_alpha`)
- E-measure (`E_phi`)
- Weighted F-measure
- Adaptive F-measure
- Mean IoU
- Mean Dice

Example:

```bash
python eval.py \
    --dataset_name "DUTS-TE" \
    --pred_path "./predictions/" \
    --gt_path "/path/to/DUTS-TE/GT/"
```

Alternatively:

```bash
bash eval.sh
```

The quantitative results will be printed directly in the terminal.

---

## Pretrained Weights and Prediction Maps

Pretrained weights, prediction maps, and related experimental files are available from Baidu Netdisk:

**Baidu Netdisk**

```text
https://pan.baidu.com/s/1gnPu6M8ZLmKG3PeGQuCczA?pwd=htf5
```

Extraction code:

```text
htf5
```

The shared files will be maintained together with the public repository.

---

## Reproducibility

The current release provides the main components required to reproduce the proposed method, including:

- Complete MASFNet network architecture
- MSPE implementation
- DCFF implementation
- CSGF implementation
- SAM2/Hiera-L backbone configuration
- Dataset loading and preprocessing
- Training pipeline
- Inference pipeline
- Quantitative evaluation pipeline
- Example shell commands
- Environment dependencies
- Pretrained weights and prediction maps

For a fair comparison, please keep the same input resolution, dataset split, and training configuration when reproducing the reported results.

Due to randomness in model optimization and data augmentation, minor numerical differences between independent runs may occur.

---

## Main Model

The complete model is implemented in:

```text
MASFNet.py
```

The overall processing pipeline can be summarized as:

```text
Input Image
    │
    ▼
SAM2 / Hiera-L Backbone
    │
    ▼
Multi-Level Features
    │
    ▼
MSPE
    │
    ▼
DCFF
    │
    ▼
CSGF-based Progressive Decoder
    │
    ▼
Segmentation Prediction
```

---

## Citation

If you find this work useful for your research, please consider citing our paper.

```bibtex
@article{MASFNet,
  title   = {Towards Accurate Image Segmentation with Multi-Scale Adaptive Feature Modeling and Semantic Fusion},
  author  = {To be updated},
  journal = {To be updated},
  year    = {2026}
}
```

The complete BibTeX information will be updated after publication.

---

## License

This repository is released under the license provided in the `LICENSE` file.

The SAM2-related implementation follows the corresponding license terms of the original SAM2 project.

---

## Acknowledgements

We thank the developers of SAM2 and the open-source research community for providing valuable implementations and resources.

---

## Updates

The repository has been updated to provide the core implementation and the main experimental pipeline required for reproducing MASFNet.

Further maintenance will focus on documentation improvements, compatibility, and additional experimental resources when necessary.
