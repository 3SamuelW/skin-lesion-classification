"""Model creation and checkpoint utilities for the timm baseline."""
from __future__ import annotations
from pathlib import Path

import torch
import timm


def create_model(architecture: str, num_classes: int,
                 pretrained: bool = True, dropout: float = 0.0):
    """Create a timm model with the given architecture and number of classes."""
    kwargs = {"pretrained": pretrained, "num_classes": num_classes}
    if dropout is not None:
        kwargs["drop_rate"] = dropout
    try:
        return timm.create_model(architecture, **kwargs)
    except TypeError:
        kwargs.pop("drop_rate", None)
        return timm.create_model(architecture, **kwargs)


def save_checkpoint(path, model, class_names, label_to_idx,
                    architecture, config, best_metric, epoch):
    """Save model weights + metadata to a single .pth file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "architecture": architecture,
        "class_names": class_names,
        "label_to_idx": label_to_idx,
        "model_state_dict": model.state_dict(),
        "best_metric": float(best_metric),
        "epoch": int(epoch),
        "timm_config": config["timm"],
    }
    torch.save(payload, path)


def load_checkpoint(path, map_location=None):
    return torch.load(path, map_location=map_location)


def load_model_from_checkpoint(path, device=None):
    """Restore the exact model used during training from a checkpoint file."""
    checkpoint = load_checkpoint(path, map_location=device)
    class_names = checkpoint["class_names"]
    architecture = checkpoint["architecture"]
    timm_cfg = checkpoint.get("timm_config", {})
    model = create_model(
        architecture,
        num_classes=len(class_names),
        pretrained=False,
        dropout=float(timm_cfg.get("dropout", 0.0)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    if device is not None:
        model = model.to(device)
    model.eval()
    return model, checkpoint
