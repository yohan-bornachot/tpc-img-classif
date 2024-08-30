from argparse import ArgumentParser
from datetime import datetime
import os
import random
import time
import pickle
import yaml

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torchvision import models
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from spoofing_faces_classifier.src.dataset import ImageDataset
from spoofing_faces_classifier.src.model import SpoofingFaceClassifier, compute_hter_metric
from spoofing_faces_classifier.src.utils import add_attr_interface, convert_to_dict


def must_save_model(cfg, epoch: int, n_epochs: int, is_best_model: bool):
    if cfg.TRAINING.SAVE_ALL_MODELS or is_best_model:
        return True
    elif epoch % cfg.TRAINING.MODEL_SAVE_PERIOD == 0 or epoch == n_epochs - 1:
        return True
    return False


def train(cfg, dataset: ImageDataset, output_root_dir: str, n_epochs: int, device: str = "cuda:0"):

    # Ensure reproducibility (seed management)
    seed = random.randint(1, 1000000000) if cfg.TRAINING.SEED is None else cfg.TRAINING.SEED
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(cfg.TRAINING.USE_DETERMINISTIC_ALGORITHMS)
    np.random.seed(seed)

    print("* Loading model and init optimizer")
    start_epoch = 0
    dt = datetime.now()
    date_dir_name = f"{dt.year}_{dt.month:02d}_{dt.day:02d}-{dt.hour:02d}h{dt.minute:02d}_{cfg.MODEL.MODEL_TYPE}"
    model = SpoofingFaceClassifier(**{k.lower(): v for k, v in cfg.MODEL.items()}).to(device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=cfg.TRAINING.LEARNING_RATE_INIT)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.TRAINING.USE_AMP)
    print(f"   -- loaded model {cfg.MODEL.MODEL_TYPE} (pretrained on {cfg.MODEL.WEIGHTS})")
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'   -- model has {total_params:,} parameters.')
    print(f"   -- model has {total_trainable_params:,} trainable parameters.")

    # Configuration and settings
    print("* Configure output")
    output_path = os.path.join(output_root_dir, date_dir_name)
    summary_path = os.path.join(output_path, 'summary')
    model_path = os.path.join(output_path, "models")
    device = torch.device(device)
    os.makedirs(model_path, exist_ok=True)
    writer = SummaryWriter(log_dir=summary_path)
    print(f"   -- output_path: {output_path}")

    # Dataloaders
    print("* Creating dataloader")
    # Look for pre-existing dataset split for specified dataset index
    split_dataset_path = os.path.join(output_root_dir, "dataset_split.pkl")
    if os.path.exists(split_dataset_path):
        with open(split_dataset_path, "rb") as pickle_file:
            split_dict = pickle.load(pickle_file)
            train_indices, val_indices = split_dict["train_indices"], split_dict["val_indices"]
            train_dataset = Subset(dataset, train_indices)
            val_dataset = Subset(dataset, val_indices)
    else:  # If not existing, create it
        train_indices, val_indices = train_test_split(np.arange(len(dataset)), test_size=1-cfg.TRAINING.VALIDATION_SPLIT,
                                                      shuffle=True, stratify=dataset.labels)
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        with open(split_dataset_path, "wb") as pickle_file:
            split_dict = {"train_indices": train_indices, "val_indices": val_indices}
            pickle.dump(split_dict, pickle_file)
    train_loader = DataLoader(train_dataset, cfg.TRAINING.BATCH_SIZE, num_workers=cfg.DATA.NB_WORKERS_TRAIN_LOADER)
    val_loader = DataLoader(val_dataset, cfg.TRAINING.BATCH_SIZE, num_workers=cfg.DATA.NB_WORKERS_VALID_LOADER)
    print(f"   -- nb batches for train: {len(train_loader)}; nb batches for validation: {len(val_loader)}")

    # Learning Rate Scheduler
    scheduler = ExponentialLR(optimizer, gamma=cfg.TRAINING.LEARNING_RATE_SCHEDULER_FACTOR)

    # Training phase
    print("* Training")
    best_hter = np.inf

    for epoch in range(start_epoch, n_epochs):
        # Epoch training
        train_losses = 0
        model.train()  # Switch to model training mode
        start_batch = time.time()
        for i_batch, (batch, labels) in enumerate(train_loader):
            # Learning
            batch = batch.to(device)
            labels = labels.to(device)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=cfg.TRAINING.USE_AMP):
                logits = model(batch).squeeze()
                train_loss = model.loss_fn(logits, labels)
            duration = time.time() - start_batch
            print(f"batch {i_batch+1}/{len(train_loader)}: training_loss={train_loss.item():.4f}\
            (duration: {duration:.2f}s)")
            writer.add_scalar("Loss/train_batch/", train_loss.item(), i_batch + epoch * len(train_loader))
            train_losses += train_loss.item()
            # Backpropagation
            scaler.scale(train_loss).backward()
            # Gradient Clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_value_(model.parameters(), cfg.TRAINING.MAX_GRADIENT_VALUE)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            start_batch = time.time()
        scheduler.step()
        epoch_train_loss = train_losses / len(train_loader)
        print(f"[Epoch {epoch+1}/{n_epochs}] - training loss = {epoch_train_loss:.4f}")
        writer.add_scalar("Loss/train/", epoch_train_loss, epoch)

        # Epoch Validation
        val_losses = 0
        preds, targets = [], []
        model.eval()  # Switch to model eval mode
        start_batch = time.time()
        for i_batch, (batch, labels) in enumerate(val_loader):
            batch = batch.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                logits = model(batch).squeeze()
                preds.append(logits.squeeze())
                targets.append(labels)
                val_loss = model.loss_fn(logits, labels)
                duration = time.time() - start_batch
                print(f"batch {i_batch + 1}/{len(val_loader)}: validation_loss={val_loss.item():.4f}\
                (duration: {duration:.2f}s)")
                val_losses += val_loss.item()
            start_batch = time.time()

        preds = torch.cat(preds)
        targets = torch.cat(targets)
        mean_val_loss = val_losses / len(val_loader)
        hter = compute_hter_metric(preds, targets)
        if hter < best_hter:
            best_hter = hter
            best_model = True
        else:
            best_model = False
        print(f"[Epoch {epoch+1}/{n_epochs}] - validation loss = {mean_val_loss:.4f}; validation HTER = {hter:.4f}")
        writer.add_scalar("Loss/validation/", mean_val_loss, epoch)
        writer.add_scalar("Metrics/HTER/", hter, epoch)

        # Saving parameters and config
        if must_save_model(cfg, epoch, n_epochs, best_model):
            model_save_dict = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch
                }
            torch.save(model_save_dict, os.path.join(model_path, f"model_epoch_{epoch}.pth"))
            with open(os.path.join(model_path, "config.yaml"), 'w') as f:
                yaml.safe_dump(convert_to_dict(cfg, []), f)
            print("   -- Saved current model")
        writer.flush()
    return model_path


if __name__ == "__main__":
    
    parser = ArgumentParser(description='Perform training of image classification model')
    parser.add_argument('--cfg_path', '-c', required=True, type=str, help='Input yaml configuration file')
    parser.add_argument('--data_dir', '-i', required=True, type=str, help='Path to data directory')
    parser.add_argument('--output_dir', '-o', required=True, type=str, help='Path to desired output location')
    parser.add_argument('--n_epochs', '-n', required=True, type=int, help='Number of epochs to perform')
    parser.add_argument('--device', '-d', default="cuda:0", type=str, help='Device to use for computations')
    args = parser.parse_args()

    with open(args.cfg_path, 'r') as f:
        cfg = add_attr_interface(yaml.safe_load(f))
        
    label_filepath = os.path.join(args.data_dir, "label_train.txt")
    train_img_dir = os.path.join(args.data_dir, "train_img")

    # Dataset instantiation
    print("* Reading dataset")
    model_transforms = getattr(models, "EfficientNet_B0" + "_Weights").DEFAULT.transforms(antialias=True)
    dataset = ImageDataset(train_img_dir, label_filepath, transform=model_transforms)
    print(f"   -- found {len(dataset)} entries")

    train(cfg, dataset, args.output_dir, args.n_epochs, args.device)
