"""Task 3: dataset loading, losses, metrics, and training loop for the U-Net.

Adapts the course lab's synthetic on-the-fly data generator into a real
Dataset class that loads the actual nuclei_dataset images/masks from disk.
Loss functions, metrics, and the training loop itself are otherwise
unchanged from the lab (same combined BCE+Dice loss, same Adam optimiser,
same 10-epoch default, same Dice/IoU definitions).
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from imaging_pipeline.data_prep import load_rgb, to_grayscale


class NucleiDataset(Dataset):
    """Loads real image/mask pairs from a nuclei_dataset split folder
    (e.g. data/nuclei_dataset/train/). Images are converted to grayscale
    (matching Task 1's preprocessing) and masks are read as binary
    (0/255 -> 0.0/1.0).
    """

    def __init__(self, split_dir: Path, augment: bool = False, seed: int = 0):
        self.split_dir = Path(split_dir)
        self.augment = augment
        self.rng = np.random.RandomState(seed)

        self.image_paths = sorted((self.split_dir / "images").glob("*.png"))
        self.mask_paths = [self.split_dir / "masks" / p.name for p in self.image_paths]
        for mp in self.mask_paths:
            if not mp.exists():
                raise FileNotFoundError(f"No matching mask for {mp}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        rgb = load_rgb(self.image_paths[idx])
        img = to_grayscale(rgb).astype(np.float32)  # already in [0, 1]

        from skimage import io
        mask_raw = io.imread(self.mask_paths[idx])
        msk = (mask_raw > 0).astype(np.float32)

        if self.augment:
            if self.rng.rand() < 0.5:
                img = np.fliplr(img).copy()
                msk = np.fliplr(msk).copy()
            if self.rng.rand() < 0.5:
                img = np.flipud(img).copy()
                msk = np.flipud(msk).copy()
            k = self.rng.randint(4)
            img = np.rot90(img, k).copy()
            msk = np.rot90(msk, k).copy()

        img_t = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)
        msk_t = torch.from_numpy(msk).unsqueeze(0)
        return img_t, msk_t


def dice_loss(logits, target, eps=1e-7):
    """Soft Dice loss in [0, 1]. Lower is better."""
    probs = torch.sigmoid(logits)
    intersection = (probs * target).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def combined_loss(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    dl = dice_loss(logits, target)
    return bce + dl


def dice_coefficient(logits, target, threshold=0.5, eps=1e-7):
    """Hard Dice for evaluation (after thresholding). Higher is better."""
    preds = (torch.sigmoid(logits) > threshold).float()
    intersection = (preds * target).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean().item()


def iou_score(logits, target, threshold=0.5, eps=1e-7):
    preds = (torch.sigmoid(logits) > threshold).float()
    intersection = (preds * target).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()


def train_unet(model, train_loader, val_loader, device, n_epochs=10, lr=1e-3, verbose=True,
                checkpoint_path=None, start_epoch=0, history=None, optimizer_state=None):
    """Train `model` for `n_epochs`, returning the training history dict
    ({"train_loss", "val_loss", "val_dice"}, one entry per epoch).

    If `checkpoint_path` is given, saves model/optimizer/history state
    after every epoch, so a long training run can be resumed with
    `start_epoch`/`history`/`optimizer_state` rather than restarted from
    scratch if interrupted.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    history = history if history is not None else {"train_loss": [], "val_loss": [], "val_dice": []}

    for epoch in range(start_epoch, n_epochs):
        model.train()
        epoch_train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = combined_loss(logits, y)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * x.size(0)
        epoch_train_loss /= len(train_loader.dataset)

        model.eval()
        epoch_val_loss = 0.0
        epoch_val_dice = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                epoch_val_loss += combined_loss(logits, y).item() * x.size(0)
                epoch_val_dice += dice_coefficient(logits, y) * x.size(0)
        epoch_val_loss /= len(val_loader.dataset)
        epoch_val_dice /= len(val_loader.dataset)

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["val_dice"].append(epoch_val_dice)

        if verbose:
            print(f"Epoch {epoch+1:2d}/{n_epochs} | "
                  f"train_loss={epoch_train_loss:.4f} | "
                  f"val_loss={epoch_val_loss:.4f} | "
                  f"val_Dice={epoch_val_dice:.4f}")

        if checkpoint_path is not None:
            torch.save({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "history": history,
            }, checkpoint_path)

    return history


def evaluate_unet(model, val_loader, device):
    """Mean Dice and IoU over the full validation set."""
    model.eval()
    total_dice, total_iou, n = 0.0, 0.0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            bs = x.size(0)
            total_dice += dice_coefficient(logits, y) * bs
            total_iou += iou_score(logits, y) * bs
            n += bs
    return total_dice / n, total_iou / n


def predict_mask(model, gray_image: np.ndarray, device) -> np.ndarray:
    """Run the U-Net on a single grayscale image, return a boolean binary
    mask (sigmoid > 0.5). Shared by Task 3's Otsu comparison and Task 4's
    hybrid pipeline so there's one inference path, not two copies of it.
    """
    model.eval()
    x = torch.from_numpy(gray_image).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        pred = (torch.sigmoid(logits) > 0.5).float().cpu().squeeze().numpy()
    return pred.astype(bool)
