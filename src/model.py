import os

import yaml
from torchvision import models
import torch

BACKBONE_OUT_FEATURES = {
    "EfficientNet_B0": 1280,
    "EfficientNet_B1": 1280,
    "EfficientNet_B2": 1408,
    "EfficientNet_B3": 1536,
    "EfficientNet_B4": 1792,
    "EfficientNet_B5": 2048,
    "EfficientNet_B6": 2304,
    "EfficientNet_B7": 2560,
    "Swin_V2_T": 768,
}


def smoothed_hter_loss(logits, targets, alpha: float = 0.5):
    """ Implementation with respect to 'AnyLoss: Transforming Classification Metrics into Loss Functions',
        Han et al. (2024)"""
    # Approximate confusion matrix with probabilities
    probs = torch.sigmoid(logits)
    tn = ((1 - targets) * (1 - probs)).sum()
    fn = (targets * (1 - probs)).sum()
    fp = ((1 - targets) * probs).sum()
    tp = (targets * probs).sum()
    # Approximates balanced accuracy
    balanced_accuracy = alpha * tn / (tn + fp) + (1 - alpha) * tp / (tp + fn)
    return 1 - balanced_accuracy  # = HTER


def compute_hter_metric(logits, targets, threshold: float = 0.5):
    outputs = torch.sigmoid(logits)
    classes = (outputs > threshold).float()
    tn = ((1 - targets) * (1 - classes)).sum()
    fn = (targets * (1 - classes)).sum()
    fp = ((1 - targets) * classes).sum()
    tp = (targets * classes).sum()
    balanced_accuracy = 1 / 2 * (tp / (tp + fn) + tn / (tn + fp))
    return 1 - balanced_accuracy  # = HTER


class SpoofingFaceClassifier(torch.nn.Module):

    def __init__(self, model_type, loss_type: str = "BCEWithLogitsLoss", alpha: float = 0.5, gammas: tuple[float] = (1., 1.), **kwargs):
        super().__init__()

        # Load pretrained model
        weights = getattr(models, model_type + "_Weights").DEFAULT
        backbone = getattr(models, model_type.lower())(weights=weights)
        backbone = torch.nn.Sequential(*list(backbone.children())[:-1])  # Remove head laye
        model = torch.nn.Module()
        model.backbone = backbone
        model.flatten = torch.nn.Flatten(start_dim=1)

        # Replace head for classification
        in_features = BACKBONE_OUT_FEATURES[model_type]
        model.head = torch.nn.Linear(in_features=in_features, out_features=1, bias=True)

        self.model = model
        self.loss_type = loss_type
        self.alpha = alpha
        self.gammas = gammas

    def forward(self, x: torch.Tensor):
        x = self.model.backbone(x)
        x = self.model.flatten(x)
        out = self.model.head(x)
        return out

    def loss_fn(self, logits: torch.Tensor, targets: torch.Tensor):
        n_pos = targets.sum()
        n_neg = len(targets) - n_pos

        if self.loss_type == "BCEWithLogitsLoss":
            return torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=n_neg/n_pos)

        elif self.loss_type == "SmoothedHTERLoss":
            return smoothed_hter_loss(logits, targets, self.alpha)

        elif self.loss_type == "MixedLoss":
            return torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=n_neg/n_pos) + \
                smoothed_hter_loss(logits, targets, self.alpha)

        else:
            raise NotImplementedError(f"loss_type {self.loss_type} is not supported")

    @classmethod
    def from_trained_torch_model(cls, model_path: str, device: str = "cpu"):
        """ Load a trained model dict from a .pth file and instantiate the corresponding SpoofingFaceClassifier"""
        saved_dict = torch.load(model_path, map_location="cpu")
        config_path = os.path.join(os.path.dirname(model_path), "config.yaml")
        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
        model = cls(**{k.lower(): v for k, v in config['MODEL'].items()})
        model.load_state_dict(saved_dict["model"])
        model = model.to(device)
        model.eval()
        return model
