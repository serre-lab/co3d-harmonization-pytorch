# Harmonization in Pytorch

This repository contains code for training deep learning models on the CO3D dataset with ClickMe 2.0, incorporating harmonization between model saliency maps and human attention data. The code is based on: https://serre-lab.github.io/Harmonization/.

## Overview

This project implements a training framework that:
- Uses the CO3D (Common Objects in 3D) dataset, but we can use any image dataset.
- Incorporates ClickMe 2.0 human clickmap data
- Provides detailed training metrics and visualization through Weights & Biases

## Installation

1. Clone the repository:
```
git clone https://github.com/serre-lab/co3d-harmonization-pytorch.git
```

2. Install the required dependencies:
```
pip install torch torchvision timm wandb numpy pillow pandas
```

## Configuration

Edit `co3d_harmonization/config.py` to set:
- Weights & Biases parameters (username, project name)
- Dataset parameters (number of classes, frame sampling rate)
- Training parameters (logging intervals)

## Usage

Train the model using the following command:
```bash
python main.py --epochs 10 --batch-size 256 --lambda-value 1.0 --ce-multiplier 1.0 --metric BCE
```

### Command Line Arguments
- `--epochs`: Number of training epochs (default: 10)
- `--batch-size`: Batch size for DataLoader (default: 256)
- `--lr`: Learning rate (default: 0.001)
- `--train-folder`: Path to the training image folder
- `--val-folder`: Path to the validation image folder
- `--lambda_value`: Weight for the harmonization loss (default: 1.0)
- `--ce_multiplier`: Multiplier for the cross-entropy loss (default: 1.0)
- `--metric`: Metric for computing harmonization loss. Options: CE, MSE, cosine, BCE (default: cosine)
- `--model`: Model name for TIMM (default: `vit_small_patch16_224.augreg_in21k_ft_in1k`)

## Monitoring

Training progress can be monitored through Weights & Biases. The following metrics are tracked:
- Validation cross-entropy loss
- Validation accuracy
- Validation alignment score (between model saliency and human attention)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
