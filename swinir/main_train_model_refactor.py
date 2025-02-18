#!/usr/bin/env python3
"""
train_with_checkpoint.py

This script trains the SwinIR model with checkpointing. Training will automatically
stop if the total elapsed time reaches a specified limit (default: 5 hours). When
this limit is reached, the current epoch is checkpointed along with the training
and validation loss history. The next run of the script can resume training from that
checkpoint using the --resume command-line argument.

Usage:
    python train_with_checkpoint.py --num_epochs 100 --max_hours 5
    python train_with_checkpoint.py --resume <checkpoint_path> --num_epochs 100 --max_hours 5
"""

import os
import time
import argparse
import datetime
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.nn import functional as F
from torchvision.models import VGG19_Weights
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor
from typing import List, Dict, Any, Tuple
from torch.cuda.amp import autocast, GradScaler

from models.network_swinir import SwinIR
from facenet_pytorch import InceptionResnetV1


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_identity_file(file_path: str) -> Tuple[List[str], List[str]]:
    """
    Load image filenames and identity labels from a text file.
    
    Each line is expected to have the format:
        <filename> <identity>
    
    Returns:
        filenames (List[str]): List of image filenames.
        ids (List[str]): List of corresponding identity labels.
    """
    with open(file_path, "r") as f:
        lines = f.read().splitlines()
    filenames = [line.split()[0].strip() for line in lines]
    ids = [line.split()[1] for line in lines]
    return filenames, ids


class CelebASRDataset(Dataset):
    """
    Dataset for CelebA Super-Resolution.
    
    Loads paired low-resolution (LR) and high-resolution (HR) images.
    """
    def __init__(self, lr_dir: str, hr_dir: str, valid_filenames: List[str]):
        """
        Args:
            lr_dir (str): Directory containing low-resolution images.
            hr_dir (str): Directory containing high-resolution images.
            valid_filenames (List[str]): List of filenames to be used (from train/test split).
        """
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        # Use only valid filenames (intersection with directory list)
        self.filenames = sorted(list(set(os.listdir(lr_dir)).intersection(valid_filenames)))
        self.transform = Compose([ToTensor()])

        if self.filenames:
            lr_sample = Image.open(os.path.join(lr_dir, self.filenames[0])).convert("RGB")
            hr_sample = Image.open(os.path.join(hr_dir, self.filenames[0])).convert("RGB")
            assert lr_sample.size == (32, 32), f"Expected LR size (32,32), got {lr_sample.size}"
            assert hr_sample.size == (128, 128), f"Expected HR size (128,128), got {hr_sample.size}"
        else:
            raise RuntimeError("No valid filenames found for dataset initialization.")

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        filename = self.filenames[idx]
        lr_path = os.path.join(self.lr_dir, filename)
        hr_path = os.path.join(self.hr_dir, filename)
        lr_img = self.transform(Image.open(lr_path).convert("RGB"))
        hr_img = self.transform(Image.open(hr_path).convert("RGB"))
        return {"lr": lr_img, "hr": hr_img, "filename": filename}


def load_swinir_model(pretrained_path: str, device: torch.device) -> nn.Module:
    """
    Load the SwinIR model with pretrained weights.
    
    If the checkpoint contains a 'params' key, that will be used; otherwise,
    the checkpoint is assumed to be the state dictionary.
    
    Args:
        pretrained_path (str): Path to the pretrained checkpoint.
        device (torch.device): Device on which to load the model.
        
    Returns:
        model (nn.Module): The loaded SwinIR model.
    """
    model = SwinIR(
        upscale=4,
        img_size=64,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="pixelshuffle",
        resi_connection="1conv",
    )
    checkpoint = torch.load(pretrained_path, map_location=device)
    state_dict = checkpoint["params"] if "params" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    return model


def perceptual_loss_fn(vgg: nn.Module, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
    """
    Compute the perceptual loss using VGG features.
    
    Args:
        vgg (nn.Module): Pretrained VGG model (feature extractor).
        sr (torch.Tensor): Super-resolved image.
        hr (torch.Tensor): Ground truth high-resolution image.
    
    Returns:
        torch.Tensor: The computed perceptual loss.
    """
    sr_features = vgg(sr)
    hr_features = vgg(hr.detach())
    return torch.mean(torch.abs(sr_features - hr_features))


def adversarial_loss_fn(fr_model: nn.Module, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
    """
    Compute the adversarial loss to fool the face recognition model.
    
    Args:
        fr_model (nn.Module): Pretrained face recognition model.
        sr (torch.Tensor): Super-resolved image.
        hr (torch.Tensor): Ground truth high-resolution image.
    
    Returns:
        torch.Tensor: The computed adversarial loss.
    """
    sr_emb = fr_model(sr)
    hr_emb = fr_model(hr)
    cos_sim = F.cosine_similarity(sr_emb, hr_emb, dim=1, eps=1e-8)
    return 1 + cos_sim.mean()


def train_model(model_gen: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer,
                pixel_loss_fn: nn.Module, vgg: nn.Module, fr_model: nn.Module,
                device: torch.device, epoch: int,
                lambda_pixel: float = 1.0, lambda_perceptual: float = 0.1, lambda_adv: float = 0.01,
                scaler: GradScaler = None) -> float:
    """
    Train the generator model for one epoch using mixed precision training.
    
    Returns:
        avg_loss (float): Average training loss for the epoch.
    """
    model_gen.train()
    if scaler is None:
        scaler = GradScaler()
    total_loss = 0.0
    num_batches = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", unit="batch")
    for batch in progress_bar:
        lr = batch["lr"].to(device)
        hr = batch["hr"].to(device)
        optimizer.zero_grad()
        
        with autocast(enabled=False):
            sr = model_gen(lr)
            loss_pixel = pixel_loss_fn(sr, hr)
            loss_perceptual = perceptual_loss_fn(vgg, sr, hr)
            loss_adv = adversarial_loss_fn(fr_model, sr, hr)
            loss = (lambda_pixel * loss_pixel +
                    lambda_perceptual * loss_perceptual +
                    lambda_adv * loss_adv)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        num_batches += 1
        progress_bar.set_postfix(loss=loss.item())
    
    avg_loss = total_loss / num_batches if num_batches else 0
    return avg_loss


def validate_model(model_gen: nn.Module, dataloader: DataLoader, pixel_loss_fn: nn.Module,
                   vgg: nn.Module, fr_model: nn.Module, device: torch.device,
                   lambda_pixel: float = 1.0, lambda_perceptual: float = 0.1, lambda_adv: float = 0.01) -> float:
    """
    Evaluate the model on the validation dataset and return the average loss.
    """
    model_gen.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Epoch Validation", unit="batch")
        for batch in progress_bar:
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)
            
            sr = model_gen(lr)
            loss_pixel = pixel_loss_fn(sr, hr)
            loss_perceptual = perceptual_loss_fn(vgg, sr, hr)
            loss_adv = adversarial_loss_fn(fr_model, sr, hr)
            loss = (lambda_pixel * loss_pixel +
                    lambda_perceptual * loss_perceptual +
                    lambda_adv * loss_adv)
            
            total_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix(loss=loss.item())
    
    avg_val_loss = total_loss / num_batches if num_batches else 0
    return avg_val_loss


def save_checkpoint(model_gen: nn.Module, optimizer: optim.Optimizer, scaler: GradScaler,
                    epoch: int, checkpoint_dir: str, train_losses: list, val_losses: list) -> str:
    """
    Save a checkpoint containing the model, optimizer, scaler states, the epoch number,
    and the loss histories.
    
    Returns:
        checkpoint_path (str): The file path of the saved checkpoint.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_filename = f"checkpoint_epoch_{epoch}_{timestamp}.pth"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
    checkpoint = {
        "epoch": epoch,
        "model_state": model_gen.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "train_losses": train_losses,
        "val_losses": val_losses
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(checkpoint_path: str, model_gen: nn.Module, optimizer: optim.Optimizer,
                    scaler: GradScaler, device: torch.device) -> Tuple[int, list, list]:
    """
    Load a checkpoint and restore the model, optimizer, scaler states, and loss histories.
    
    Returns:
        start_epoch (int): The epoch from which to resume training.
        train_losses (list): Previously logged training losses.
        val_losses (list): Previously logged validation losses.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_gen.load_state_dict(checkpoint["model_state"], strict=False)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scaler.load_state_dict(checkpoint["scaler_state"])
    start_epoch = checkpoint["epoch"]
    print(f"Resuming training from epoch {start_epoch + 1}")
    train_losses = checkpoint.get("train_losses", [])
    val_losses = checkpoint.get("val_losses", [])
    return start_epoch, train_losses, val_losses


def main():
    parser = argparse.ArgumentParser(description="Train SwinIR with checkpointing")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training")
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="Total number of epochs to train")
    parser.add_argument("--max_hours", type=float, default=5,
                        help="Maximum allowed training time in hours per run")
    args = parser.parse_args()

    # ----------------------
    # Setup
    # ----------------------
    os.makedirs("output", exist_ok=True)
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Directories for datasets
    lr_dir = "./datasets/celeba_LR_factor_0.25"
    hr_dir = "./datasets/celeba_HR_resized_128"

    # Load identity splits for training
    train_img_filenames, train_img_ids = load_identity_file("img_processing/split_ids/identity_train.txt")
    print(f"Training identities: {len(set(train_img_ids))} | Training images: {len(train_img_filenames)}")
    print("Loading CelebA-SR training dataset...")
    train_set = CelebASRDataset(lr_dir, hr_dir, valid_filenames=train_img_filenames)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=20, pin_memory=True)
    print(f"Dataset loaded with {len(train_set)} training samples.")

    # Load identity splits for validation
    val_img_filenames, val_img_ids = load_identity_file("img_processing/split_ids/identity_val.txt")
    print(f"Validation identities: {len(set(val_img_ids))} | Validation images: {len(val_img_filenames)}")
    print("Loading CelebA-SR validation dataset...")
    val_set = CelebASRDataset(lr_dir, hr_dir, valid_filenames=val_img_filenames)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=20, pin_memory=True)
    print(f"Dataset loaded with {len(val_set)} validation samples.")

    # ----------------------
    # Model, Optimizer, and Scaler Setup
    # ----------------------
    swinir_path = "./model_zoo/swinir/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth"
    model_gen = load_swinir_model(swinir_path, device).to(device)

    # Freeze early layers to preserve low-level features.
    for name, param in model_gen.named_parameters():
        if "layers.0" in name or "patch_embed" in name:
            param.requires_grad = False
    print("SwinIR model loaded successfully.")

    # Load Face Recognition model (for adversarial loss)
    fr_model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    for param in fr_model.parameters():
        param.requires_grad = False

    # VGG model for perceptual loss (using first 16 layers)
    vgg_full = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).to(device)
    vgg = nn.Sequential(*list(vgg_full.features.children())[:16]).eval()
    for param in vgg.parameters():
        param.requires_grad = False

    # Loss function and optimizer
    pixel_loss_fn = nn.L1Loss().to(device)
    optimizer = optim.Adam(model_gen.parameters(), lr=1e-5)
    scaler = GradScaler()

    # Initialize loss histories.
    start_epoch = 0
    train_losses = []
    val_losses = []
    if args.resume is not None:
        start_epoch, train_losses, val_losses = load_checkpoint(args.resume, model_gen, optimizer, scaler, device)

    # Maximum allowed training time (in seconds)
    max_training_time = args.max_hours * 3600
    training_start_time = time.time()

    # Loss weights
    pixel_w = 1.0
    perceptual_w = 0.1
    adv_w = 0.01

    print("Starting training...")
    for epoch in range(start_epoch + 1, args.num_epochs + 1):
        print(f"\nStarting epoch {epoch}")
        # Training step
        train_loss = train_model(
            model_gen,
            train_loader,
            optimizer,
            pixel_loss_fn,
            vgg,
            fr_model,
            device,
            epoch,
            lambda_pixel=pixel_w,
            lambda_perceptual=perceptual_w,
            lambda_adv=adv_w,
            scaler=scaler,
        )
        print(f"Epoch {epoch} Training Loss: {train_loss:.4f}")
        train_losses.append(train_loss)
        
        # Validation step
        val_loss = validate_model(
            model_gen,
            val_loader,
            pixel_loss_fn,
            vgg,
            fr_model,
            device,
            lambda_pixel=pixel_w,
            lambda_perceptual=perceptual_w,
            lambda_adv=adv_w,
        )
        print(f"Epoch {epoch} Validation Loss: {val_loss:.4f}")
        val_losses.append(val_loss)

        # Check elapsed time for checkpointing.
        elapsed_time = time.time() - training_start_time
        if elapsed_time >= max_training_time:
            print(f"Elapsed training time {elapsed_time/3600:.2f} hours reached limit of {args.max_hours} hours.")
            save_checkpoint(model_gen, optimizer, scaler, epoch, checkpoint_dir="output",
                            train_losses=train_losses, val_losses=val_losses)
            print(f"Stopping training now at epoch {epoch}. Please resume later.")
            return

    # Training complete: save the final model.
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    final_model_path = os.path.join("output", f"swinir_fr_model_{timestamp}.pth")
    torch.save(model_gen.state_dict(), final_model_path)
    print(f"Training completed. Final model saved at {final_model_path}")
    
    # Save final loss histories to a file.
    loss_history = {"train_losses": train_losses, "val_losses": val_losses}
    loss_file = os.path.join("output", f"loss_history_{timestamp}.json")
    with open(loss_file, "w") as f:
        json.dump(loss_history, f, indent=4)
    print(f"Loss history saved at {loss_file}")


if __name__ == "__main__":
    main()
