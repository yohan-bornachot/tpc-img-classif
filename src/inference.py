import os
from argparse import ArgumentParser

import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import models

from spoofing_faces_classifier.src.dataset import ImageDataset
from spoofing_faces_classifier.src.model import SpoofingFaceClassifier
from spoofing_faces_classifier.src.utils import add_attr_interface


def infer(test_dataset: ImageDataset, model: SpoofingFaceClassifier, batch_size: int = 256, device: str = 'cpu'):
    test_loader = DataLoader(test_dataset, batch_size, num_workers=cfg.DATA.NB_WORKERS_VALID_LOADER)

    predicted_classes = []
    model.eval()  # Switch to model eval mode
    for i_batch, (batch, _) in enumerate(test_loader):
        batch = batch.to(device)
        with torch.no_grad():
            logits = model(batch).squeeze()
            # Convert logits to probabilities
            if len(logits.shape) == 2 and logits.shape[1] == 2:
                outputs = torch.softmax(logits, dim=1)
                classes = torch.argmax(outputs, dim=1).int()
            else:
                outputs = torch.sigmoid(logits)
                classes = (outputs > 0.5).int()
            predicted_classes += classes.tolist()
            print(f"Processed batch {i_batch + 1}/{len(test_loader)}")
    return predicted_classes


def write_results(predicted_classes: list[int], output_filepath: str):
    with open(output_filepath, "w") as output_file:
        for pred in predicted_classes:
            output_file.write(f"{str(pred)}\n")


if __name__ == "__main__":
    parser = ArgumentParser(description='Perform inference on test data')
    parser.add_argument('--model_path', '-m', required=True, type=str, help='Path to trained model')
    parser.add_argument('--data_dir', '-i', required=True, type=str, help='Path to data directory')
    parser.add_argument('--output_dir', '-o', required=True, type=str, help='Path to desired output location')
    parser.add_argument('--device', '-d', default="cuda:0", type=str, help='Device to use for computations')
    args = parser.parse_args()

    # Load trained model
    model = SpoofingFaceClassifier.from_trained_torch_model(args.model_path, device=args.device)

    # Load config which lies along trained model
    cfg_path = os.path.join(os.path.dirname(args.model_path), "config.yaml")
    with open(cfg_path, 'r') as f:
        cfg = add_attr_interface(yaml.safe_load(f))

    test_img_dir = os.path.join(args.data_dir, "val_img")

    # Dataset instantiation
    print("* Reading dataset")
    model_transforms = getattr(models, cfg.MODEL.MODEL_TYPE + "_Weights").DEFAULT.transforms(antialias=True)
    test_dataset = ImageDataset(test_img_dir, transform=model_transforms)
    print(f"   -- found {len(test_dataset)} entries")

    preds = infer(test_dataset, model, cfg.TRAINING.BATCH_SIZE, args.device)

    output_filepath = os.path.join(args.output_dir, "label_test.txt")
    write_results(preds, output_filepath)
