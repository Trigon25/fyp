#!/usr/bin/env python3
"""
evaluate_model.py

This script loads a pretrained SwinIR model from disk and evaluates it on the
test split of the CelebA-SR dataset. It computes L1 loss, PSNR, SSIM, and the
fooling rate (using a face recognition model). Metrics are printed to the console
and saved to "output/eval_metrics.txt".

Usage:
    python evaluate_model.py
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor
from PIL import Image
from tqdm import tqdm
from skimage.metrics import structural_similarity as compute_ssim
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
import torchvision.models as models
from torchvision.models import VGG19_Weights

from models.network_swinir import SwinIR
from facenet_pytorch import InceptionResnetV1
from datetime import datetime


def load_identity_file(file_path: str):
    """
    Load image filenames and identities from a text file.
    
    Each line should have the format:
        <filename> <identity>
    
    Returns:
        filenames (List[str]): List of image filenames.
        ids (List[str]): List of identity labels.
    """
    with open(file_path, "r") as f:
        lines = f.read().splitlines()
    filenames = [line.split()[0].strip() for line in lines]
    ids = [line.split()[1] for line in lines]
    return filenames, ids


class CelebASRDataset(Dataset):
    """
    Dataset for CelebA Super-Resolution evaluation.
    
    Expects paired low-resolution (LR) and high-resolution (HR) images.
    """
    def __init__(self, lr_dir: str, hr_dir: str, valid_filenames: list):
        """
        Args:
            lr_dir (str): Directory with LR images.
            hr_dir (str): Directory with HR images.
            valid_filenames (list): List of filenames to use (from test split).
        """
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        # Only use images that exist in lr_dir and are part of valid_filenames.
        self.filenames = sorted(list(set(os.listdir(lr_dir)).intersection(valid_filenames)))
        self.transform = Compose([ToTensor()])
        
        if self.filenames:
            # Validate image dimensions (expect LR: 32x32, HR: 128x128)
            lr_sample = Image.open(os.path.join(lr_dir, self.filenames[0])).convert("RGB")
            hr_sample = Image.open(os.path.join(hr_dir, self.filenames[0])).convert("RGB")
            assert lr_sample.size == (32, 32), f"Expected LR size (32,32), got {lr_sample.size}"
            assert hr_sample.size == (128, 128), f"Expected HR size (128,128), got {hr_sample.size}"
        else:
            raise RuntimeError("No valid filenames found for dataset initialization.")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx: int):
        filename = self.filenames[idx]
        lr_path = os.path.join(self.lr_dir, filename)
        hr_path = os.path.join(self.hr_dir, filename)
        lr_img = Image.open(lr_path).convert("RGB")
        hr_img = Image.open(hr_path).convert("RGB")
        lr_tensor = self.transform(lr_img)
        hr_tensor = self.transform(hr_img)
        return {"lr": lr_tensor, "hr": hr_tensor, "filename": filename}


def load_swinir_model(pretrained_path: str, device: torch.device):
    """
    Instantiate the SwinIR model and load pretrained weights.
    
    Args:
        pretrained_path (str): Path to the saved model checkpoint.
        device (torch.device): Device for model loading.
        
    Returns:
        model (nn.Module): Loaded SwinIR model.
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


def calculate_psnr_tensor(sr: torch.Tensor, hr: torch.Tensor) -> float:
    """
    Calculate PSNR for a pair of images.
    """
    mse = torch.mean((sr - hr) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10 * np.log10(1.0 / mse)


def evaluate_model(model_gen: nn.Module, dataloader: DataLoader, pixel_loss_fn: nn.Module, device: torch.device):
    """
    Evaluate the model by computing L1 loss, PSNR, and SSIM.
    
    Args:
        model_gen (nn.Module): The super-resolution generator model.
        dataloader (DataLoader): Test data loader.
        pixel_loss_fn (nn.Module): Loss function (e.g., L1Loss).
        device (torch.device): Device for evaluation.
        
    Returns:
        metrics (dict): Averaged L1, PSNR, and SSIM metrics.
    """
    model_gen.eval()
    metrics = {"L1": 0.0, "PSNR": 0.0, "SSIM": 0.0}
    total_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", unit="batch"):
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)
            sr = model_gen(lr)

            metrics["L1"] += pixel_loss_fn(sr, hr).item()
            metrics["PSNR"] += calculate_psnr_tensor(sr, hr)

            # Convert tensors to NumPy arrays in (H, W, C) format.
            sr_cpu = sr.cpu().numpy()
            hr_cpu = hr.cpu().numpy()
            batch_ssim = 0.0
            for i in range(sr_cpu.shape[0]):
                sr_img = np.transpose(sr_cpu[i], (1, 2, 0))
                hr_img = np.transpose(hr_cpu[i], (1, 2, 0))
                # Note: Use channel_axis=-1 for images in (H, W, C) format.
                batch_ssim += compute_ssim(hr_img, sr_img, data_range=1.0, channel_axis=-1)
            metrics["SSIM"] += batch_ssim / sr_cpu.shape[0]
            total_batches += 1

    # Average metrics over batches.
    for key in metrics:
        metrics[key] /= total_batches

    return metrics


def evaluate_fooling_rate(
    generator: nn.Module,
    fr_model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.8,
):
    """
    Evaluate the fooling rate of the generator (i.e., the rate at which the face
    recognition model misclassifies the super-resolved images compared to the HR images).

    Args:
        generator (nn.Module): The super-resolution generator.
        fr_model (nn.Module): Face recognition model.
        dataloader (DataLoader): Data loader for the test set.
        device (torch.device): Device for evaluation.
        threshold (float): Cosine similarity threshold below which the SR image is considered
                           to have a different (wrong) identity.

    Returns:
        fooling_rate (float): Fraction of images where SR predictions differ from HR.
    """
    generator.eval()
    fr_model.eval()

    fooled_count = 0
    samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Fooling Rate", unit="batch"):
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)
            sr = generator(lr)

            emb_hr = fr_model(hr)
            emb_sr = fr_model(sr)

            similarity = F.cosine_similarity(emb_hr, emb_sr, dim=1)

            fooled_count += (similarity < threshold).sum().item()
            total_samples += hr.size(0)

    return fooled_count / total_samples if total_samples > 0 else 0.0


def main():
    # -------------------------
    # Arguments
    # -------------------------
        # Directories and file paths (adjust these paths as needed)
    parser = argparse.ArgumentParser(description="Evaluate SwinIR model on CelebA-SR dataset")
    parser.add_argument("--lr_dir", type=str, default="./datasets/celeba_LR_factor_0.25", help="Directory with LR images")
    parser.add_argument("--hr_dir", type=str, default="./datasets/celeba_HR_resized_128", help="Directory with HR images")
    parser.add_argument("--test_identity_file", type=str, default="img_processing/split_ids/identity_test.txt", help="Path to the test identity file")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the saved SwinIR model checkpoint")
    args = parser.parse_args()

    lr_dir = args.lr_dir
    hr_dir = args.hr_dir
    test_identity_file = args.test_identity_file
    model_path = args.model_path
    
    # -------------------------
    # Setup
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -------------------------
    # Prepare Test Dataset
    # -------------------------
    test_img_filenames, _ = load_identity_file(test_identity_file)
    test_set = CelebASRDataset(lr_dir, hr_dir, valid_filenames=test_img_filenames)
    test_loader = DataLoader(
        test_set, batch_size=64, shuffle=False, num_workers=20, pin_memory=True
    )
    print(f"Loaded test set with {len(test_set)} images.")

    # -------------------------
    # Load Models
    # -------------------------
    print("Loading saved SwinIR model...")
    if model_path is None:
        raise ValueError("Model path must be specified.")
    model_gen = load_swinir_model(model_path, device).to(device)

    # Load Face Recognition model (for fooling rate evaluation)
    fr_model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    for param in fr_model.parameters():
        param.requires_grad = False

    # Loss function for L1 loss
    pixel_loss_fn = nn.L1Loss().to(device)

    # -------------------------
    # Evaluation
    # -------------------------
    print("Evaluating model performance on test set...")
    metrics = evaluate_model(model_gen, test_loader, pixel_loss_fn, device)
    fooling_rate = evaluate_fooling_rate(model_gen, fr_model, test_loader, device)

    print("\nEvaluation Metrics:")
    print(f"  L1 Loss: {metrics['L1']:.4f}")
    print(f"  PSNR:    {metrics['PSNR']:.2f} dB")
    print(f"  SSIM:    {metrics['SSIM']:.4f}")
    print(f"  Fooling Rate: {fooling_rate:.4f}")

    # Save the evaluation metrics to a file.
    os.makedirs("output", exist_ok=True)
    model_name = os.path.basename(model_path).split('.')[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_file = os.path.join("output", f"eval_metrics_{model_name}_{timestamp}.txt")
    with open(metrics_file, "w") as f:
        f.write(f"L1: {metrics['L1']:.4f}\n")
        f.write(f"PSNR: {metrics['PSNR']:.2f}\n")
        f.write(f"SSIM: {metrics['SSIM']:.4f}\n")
        f.write(f"Fooling Rate (fooled/total): {fooling_rate:.4f}\n")
    print(f"Evaluation metrics saved to '{metrics_file}'.")


if __name__ == "__main__":
    main()
