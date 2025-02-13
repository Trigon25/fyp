#!/usr/bin/env python3
"""
validate_model.py

This script loads a pretrained SwinIR model from disk and processes the 
validation split of the CelebA-SR dataset. It uses the filenames from 
identity_val.txt to select images, runs the low-resolution images through 
the model, and saves the resulting super-resolved images to the specified 
output directory.

Usage:
    python validate_model.py --model_path <path_to_checkpoint>
"""

import os
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, ToPILImage
from PIL import Image
from tqdm import tqdm

from datetime import datetime
from models.network_swinir import SwinIR


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
    Dataset for CelebA Super-Resolution validation.
    
    Expects paired low-resolution (LR) and high-resolution (HR) images.
    """
    def __init__(self, lr_dir: str, hr_dir: str, valid_filenames: list):
        """
        Args:
            lr_dir (str): Directory with LR images.
            hr_dir (str): Directory with HR images.
            valid_filenames (list): List of filenames to use (from validation split).
        """
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        # Use only images that are both in the LR directory and in the valid filenames.
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


def validate_model(model_gen: torch.nn.Module, dataloader: DataLoader, output_dir: str, device: torch.device):
    """
    Run the model on the validation set and save super-resolved images.
    
    Args:
        model_gen (torch.nn.Module): The super-resolution generator model.
        dataloader (DataLoader): Validation data loader.
        output_dir (str): Directory to save the super-resolved images.
        device (torch.device): Device for evaluation.
    """
    model_gen.eval()
    to_pil = ToPILImage()
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", unit="batch"):
            lr = batch["lr"].to(device)
            filenames = batch["filename"]
            sr = model_gen(lr)
            # Iterate over each image in the batch.
            for i in range(sr.size(0)):
                # Clamp values to [0,1] and convert tensor to a PIL image.
                sr_img_tensor = sr[i].cpu().clamp(0, 1)
                sr_img = to_pil(sr_img_tensor)
                output_path = os.path.join(output_dir, filenames[i])
                sr_img.save(output_path)


def main():
    parser = argparse.ArgumentParser(description="Validate SwinIR model on CelebA-SR validation set")
    parser.add_argument("--lr_dir", type=str, default="./datasets/celeba_LR_factor_0.25",
                        help="Directory with LR images")
    parser.add_argument("--hr_dir", type=str, default="./datasets/celeba_HR_resized_128",
                        help="Directory with HR images")
    parser.add_argument("--val_identity_file", type=str, default="img_processing/split_ids/identity_val.txt",
                        help="Path to the validation identity file")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the saved SwinIR model checkpoint")
    parser.add_argument("--output_dir", type=str, default="output/val_results",
                        help="Directory to save the super-resolved validation images")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for processing")
    parser.add_argument("--num_workers", type=int, default=16,
                        help="Number of worker threads for DataLoader")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the validation identity file.
    val_img_filenames, _ = load_identity_file(args.val_identity_file)
    print(f"Loaded {len(val_img_filenames)} filenames from {args.val_identity_file}")

    # Create the validation dataset and dataloader.
    val_set = CelebASRDataset(args.lr_dir, args.hr_dir, valid_filenames=val_img_filenames)
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"Validation dataset contains {len(val_set)} images.")

    # Load the SwinIR model.
    print("Loading SwinIR model...")
    model_gen = load_swinir_model(args.model_path, device).to(device)

    # Process the validation set and save the super-resolved images.
    print("Running validation and saving super-resolved images...")
    validate_model(model_gen, val_loader, args.output_dir, device)
    print(f"Super-resolved validation images saved to '{args.output_dir}'.")


if __name__ == "__main__":
    main()
