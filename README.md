# Spoofing Faces Classifier

This repository contains a deep learning project designed to classify images for face spoofing detection. The project uses PyTorch and includes modules for training, inference, and evaluation of a custom model. 

## Table of Contents
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Configuration](#configuration)
- [Training](#training)
- [Inference](#inference)
- [Results](#results)

## Project Structure

```plaintext
.
|-- README.md
|-- TPC-img-classif.ipynb  # Jupyter Notebook for exploratory analysis and prototyping
|-- config.yml             # Configuration file for training and inference
|-- data                   # Directory containing datasets
|   |-- label_test.txt     # Labels for the test dataset
|   |-- label_train.txt    # Labels for the training dataset
|   |-- train_img          # Directory containing training images
|   `-- val_img            # Directory containing validation images
|-- output                 # Directory for storing model outputs, logs, etc.
|-- requirements.txt       # Needed libraries
`-- src                    # Source code for the project
    |-- dataset.py         # Dataset class for loading and processing images
    |-- inference.py       # Script for performing inference
    |-- model.py           # Definition of the classifier model
    |-- train.py           # Training script
    `-- utils.py           # Utility functions
```

## Requirements

- Python 3.8+
- PyTorch 1.10+
- torchvision 0.11+
- numpy
- pyyaml
- scikit-learn
- Pillow

You can install the necessary dependencies using pip:

```bash
pip install -r requirements.txt
```

## Configuration

The `config.yml` file contains all the necessary configurations for training and inference. Key sections include:

- `DATA`: Parameters related to data loading and processing.
- `MODEL`: Model architecture and loss function configuration.
- `TRAINING`: Hyperparameters and training settings.

You can modify the `config.yml` file to fit your specific requirements.

## Training

To train the model, use the following command:

```bash
python src/train.py --cfg_path config.yml --data_dir ./data --output_dir ./output --n_epochs 50 --device cuda:0
```

- `--cfg_path`: Path to the configuration file.
- `--data_dir`: Directory containing the training and validation datasets.
- `--output_dir`: Directory to save model outputs and logs.
- `--n_epochs`: Number of epochs to train.
- `--device`: Device to use for training (e.g., `cuda:0` for GPU, `cpu` for CPU).

During training, model checkpoints and TensorBoard logs will be saved in the specified output directory.

## Inference

To perform inference on a new dataset, run:

```bash
python src/inference.py --model_path ./output/model_epoch_49.pth --data_dir ./data --output_dir ./output --device cuda:0
```

- `--model_path`: Path to the trained model file.
- `--data_dir`: Directory containing the images for inference.
- `--output_dir`: Directory to save the inference results.
- `--device`: Device to use for inference.

The results will be saved in `label_test.txt` in the output directory.

## Results

After training, the metrics such as loss, and Half-Total Error Rate (HTER) will be logged. You can visualize these metrics using TensorBoard:

```bash
tensorboard --logdir=./output/summary
```
