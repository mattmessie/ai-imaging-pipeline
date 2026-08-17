"""
scripts/run_task3_unet.py

Task 3: train the U-Net on the real nuclei dataset, evaluate with mean
Dice/IoU on the held-out validation split, and visualize input/ground-
truth/prediction for several validation images.

Runs entirely locally with no external LLM dependency -- no Ollama, no
Colab workaround needed for this task.

Saves:
    outputs/figures/task3_training_curves.png
    outputs/figures/task3_predictions.png
    outputs/metrics/task3_dice_iou.txt
    outputs/model_objects/unet_trained.pth
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from imaging_pipeline.config import TRAIN_DIR, VAL_DIR, FIGURE_DIR, METRICS_DIR, OUTPUT_DIR
from imaging_pipeline.unet import UNet
from imaging_pipeline.train_unet import NucleiDataset, train_unet, evaluate_unet

MODEL_DIR = OUTPUT_DIR / "model_objects"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = MODEL_DIR / "unet_checkpoint.pth"

RANDOM_SEED = 42
N_EPOCHS = 10


def plot_training_curves(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["train_loss"], label="train_loss")
    axes[0].plot(history["val_loss"], label="val_loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Combined loss (BCE + Dice)")
    axes[0].set_title("Training and validation loss")
    axes[0].legend()

    axes[1].plot(history["val_dice"], color="green", marker="o")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation Dice")
    axes[1].set_title("Validation Dice over training")
    axes[1].set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_predictions(model, val_ds, device, save_path, n_show=4):
    model.eval()
    fig, axes = plt.subplots(n_show, 3, figsize=(9, 3 * n_show))
    with torch.no_grad():
        for i in range(n_show):
            x, y = val_ds[i]
            x_dev = x.unsqueeze(0).to(device)
            logits = model(x_dev)
            pred = (torch.sigmoid(logits) > 0.5).float().cpu().squeeze().numpy()

            axes[i, 0].imshow(x.squeeze().numpy(), cmap="gray")
            axes[i, 0].set_title(f"Input (val #{i})")
            axes[i, 0].axis("off")
            axes[i, 1].imshow(y.squeeze().numpy(), cmap="gray")
            axes[i, 1].set_title("Ground truth")
            axes[i, 1].axis("off")
            axes[i, 2].imshow(pred, cmap="gray")
            axes[i, 2].set_title("U-Net prediction")
            axes[i, 2].axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {device}")

    train_ds = NucleiDataset(TRAIN_DIR, augment=True, seed=0)
    val_ds = NucleiDataset(VAL_DIR, augment=False)
    print(f"Train: {len(train_ds)} images, Val: {len(val_ds)} images")

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    model = UNet(in_ch=1, out_ch=1, base=16).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"U-Net created. Parameter count: {n_params:,}")

    # Resume from checkpoint if a previous (possibly interrupted) run left one.
    start_epoch, history, optimizer_state = 0, None, None
    if CHECKPOINT_PATH.exists():
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        start_epoch = ckpt["epoch"]
        history = ckpt["history"]
        optimizer_state = ckpt["optimizer_state"]
        print(f"Resuming from checkpoint at epoch {start_epoch}/{N_EPOCHS}")

    print(f"\nTraining for {N_EPOCHS} epochs (starting from epoch {start_epoch})...")
    t0 = time.time()
    history = train_unet(
        model, train_loader, val_loader, device, n_epochs=N_EPOCHS,
        checkpoint_path=CHECKPOINT_PATH, start_epoch=start_epoch,
        history=history, optimizer_state=optimizer_state,
    )
    print(f"Training complete in {time.time()-t0:.1f}s")

    plot_training_curves(history, FIGURE_DIR / "task3_training_curves.png")

    print("\nFinal evaluation on full validation set...")
    mean_dice, mean_iou = evaluate_unet(model, val_loader, device)
    print(f"Validation Dice: {mean_dice:.4f}")
    print(f"Validation IoU:  {mean_iou:.4f}")

    with open(METRICS_DIR / "task3_dice_iou.txt", "w") as f:
        f.write(f"U-Net training (base=16, {N_EPOCHS} epochs, combined BCE+Dice loss, Adam lr=1e-3)\n")
        f.write(f"Parameter count: {n_params:,}\n\n")
        f.write(f"Final validation Dice: {mean_dice:.4f}\n")
        f.write(f"Final validation IoU:  {mean_iou:.4f}\n\n")
        f.write("Per-epoch history:\n")
        for i in range(N_EPOCHS):
            f.write(f"  Epoch {i+1:2d}: train_loss={history['train_loss'][i]:.4f}, "
                     f"val_loss={history['val_loss'][i]:.4f}, val_dice={history['val_dice'][i]:.4f}\n")

    print("\nSaving prediction visualizations for 4 validation images...")
    plot_predictions(model, val_ds, device, FIGURE_DIR / "task3_predictions.png", n_show=4)

    torch.save(model.state_dict(), MODEL_DIR / "unet_trained.pth")
    print(f"\nSaved trained weights to {MODEL_DIR / 'unet_trained.pth'}")
    print(f"Saved training curves to {FIGURE_DIR / 'task3_training_curves.png'}")
    print(f"Saved predictions to {FIGURE_DIR / 'task3_predictions.png'}")
    print(f"Saved metrics to {METRICS_DIR / 'task3_dice_iou.txt'}")


if __name__ == "__main__":
    main()
