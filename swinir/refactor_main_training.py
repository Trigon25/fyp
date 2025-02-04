import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision.models import VGG19_Weights
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
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

        # Validate image sizes using the first sample
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
    model.load_state_dict(checkpoint["params"], strict=False)
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
    return 1 - torch.nn.functional.cosine_similarity(sr_emb, hr_emb).mean()


def calculate_psnr_tensor(sr: torch.Tensor, hr: torch.Tensor) -> float:
    """
    Calculate PSNR (Peak Signal-to-Noise Ratio) for two images.
    
    Args:
        sr (torch.Tensor): Super-resolved image.
        hr (torch.Tensor): High-resolution image.
    
    Returns:
        float: The PSNR value.
    """
    mse = torch.mean((sr - hr) ** 2).item()
    if mse == 0:
        return float('inf')
    return 10 * np.log10(1.0 / mse)


def evaluate_fooling_rate(generator: nn.Module, fr_model: nn.Module,
                            dataloader: DataLoader, device: torch.device) -> float:
    """
    Evaluate the fooling rate (the rate at which the generator fools the FR model).
    
    Note: This implementation uses a naive approach by taking argmax over the FR embeddings.
    
    Args:
        generator (nn.Module): The super-resolution generator model.
        fr_model (nn.Module): The face recognition model.
        dataloader (DataLoader): DataLoader for the test set.
        device (torch.device): Device to run evaluation on.
    
    Returns:
        float: The fooling rate (higher is better).
    """
    correct = 0
    total = 0
    generator.eval()
    with torch.no_grad():
        for batch in dataloader:
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)
            sr = generator(lr)
            # InceptionResnetV1 outputs embeddings, so argmax is only a proxy here.
            pred_hr = fr_model(hr).argmax(dim=1)
            pred_sr = fr_model(sr).argmax(dim=1)
            total += hr.size(0)
            correct += (pred_sr != pred_hr).sum().item()
    generator.train()
    return correct / total if total > 0 else 0.0


def train_model(model_gen: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer,
                pixel_loss_fn: nn.Module, vgg: nn.Module, fr_model: nn.Module,
                device: torch.device, epoch: int,
                lambda_pixel: float = 1.0, lambda_perceptual: float = 0.1, lambda_adv: float = 0.01,
                scaler: GradScaler = None) -> None:
    """
    Train the generator model for one epoch using mixed precision training.
    
    Args:
        model_gen (nn.Module): Generator model.
        dataloader (DataLoader): Training data loader.
        optimizer (optim.Optimizer): Optimizer.
        pixel_loss_fn (nn.Module): Pixel-wise loss function (e.g., L1).
        vgg (nn.Module): VGG model for perceptual loss.
        fr_model (nn.Module): Face recognition model for adversarial loss.
        device (torch.device): Training device.
        epoch (int): Current epoch number.
        lambda_pixel (float): Weight for pixel loss.
        lambda_perceptual (float): Weight for perceptual loss.
        lambda_adv (float): Weight for adversarial loss.
        scaler (GradScaler): AMP GradScaler instance.
    """
    model_gen.train()
    if scaler is None:
        scaler = GradScaler()
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}", unit="batch")
    for batch in progress_bar:
        lr = batch["lr"].to(device)
        hr = batch["hr"].to(device)
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        with autocast():
            sr = model_gen(lr)
            loss_pixel = pixel_loss_fn(sr, hr)
            loss_perceptual = perceptual_loss_fn(vgg, sr, hr)
            loss_adv = adversarial_loss_fn(fr_model, sr, hr)
            total_loss = (lambda_pixel * loss_pixel +
                          lambda_perceptual * loss_perceptual +
                          lambda_adv * loss_adv)
        
        # Backpropagation with gradient scaling
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        progress_bar.set_postfix(loss=total_loss.item())


def evaluate_model(model_gen: nn.Module, dataloader: DataLoader,
                   pixel_loss_fn: nn.Module, device: torch.device) -> Dict[str, float]:
    """
    Evaluate the generator model on the test set.
    
    Computes average L1 loss, PSNR, and SSIM.
    
    Args:
        model_gen (nn.Module): Generator model.
        dataloader (DataLoader): Test data loader.
        pixel_loss_fn (nn.Module): L1 loss function.
        device (torch.device): Evaluation device.
    
    Returns:
        Dict[str, float]: Dictionary containing averaged metrics.
    """
    model_gen.eval()
    metrics = {"L1": 0.0, "PSNR": 0.0, "SSIM": 0.0}
    total_batches = 0
    with torch.no_grad():
        for batch in dataloader:
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)
            sr = model_gen(lr)

            metrics["L1"] += pixel_loss_fn(sr, hr).item()
            metrics["PSNR"] += calculate_psnr_tensor(sr, hr)

            # Compute SSIM per image in the batch
            sr_cpu = sr.cpu().numpy()
            hr_cpu = hr.cpu().numpy()
            batch_ssim = 0.0
            for i in range(sr_cpu.shape[0]):
                # Convert from (C, H, W) to (H, W, C)
                sr_img = np.transpose(sr_cpu[i], (1, 2, 0))
                hr_img = np.transpose(hr_cpu[i], (1, 2, 0))
                batch_ssim += compute_ssim(hr_img, sr_img, data_range=1.0, multichannel=True)
            metrics["SSIM"] += batch_ssim / sr_cpu.shape[0]
            total_batches += 1

    # Average metrics over all batches
    for key in metrics:
        metrics[key] /= total_batches
    return metrics


def main():
    # ----------------------
    # Setup
    # ----------------------
    os.makedirs("output", exist_ok=True)
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Directories for low-resolution and high-resolution images
    lr_dir = "./datasets/celeba_LR_factor_0.25"
    hr_dir = "./datasets/celeba_HR_resized_128"

    # Load identity splits
    train_img_filenames, train_img_ids = load_identity_file("img_processing/split_ids/identity_train.txt")
    test_img_filenames, test_img_ids = load_identity_file("img_processing/split_ids/identity_test.txt")
    print(f"Training identities: {len(set(train_img_ids))} | Training images: {len(train_img_filenames)}")
    print(f"Test identities: {len(set(test_img_ids))} | Test images: {len(test_img_filenames)}")

    # ----------------------
    # Dataset Preparation
    # ----------------------
    print("Loading CelebA-SR dataset...")
    train_set = CelebASRDataset(lr_dir, hr_dir, valid_filenames=train_img_filenames)
    test_set = CelebASRDataset(lr_dir, hr_dir, valid_filenames=test_img_filenames)

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)
    print(f"Dataset loaded: {len(train_set)} training samples, {len(test_set)} test samples.")

    # ----------------------
    # Model Loading
    # ----------------------
    print("Loading SwinIR model...")
    swinir_path = "./model_zoo/swinir/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth"
    model_gen = load_swinir_model(swinir_path, device).to(device)

    # Freeze early layers to preserve low-level features
    for name, param in model_gen.named_parameters():
        if "layers.0" in name or "patch_embed" in name:
            param.requires_grad = False
    print("SwinIR model loaded successfully.")

    # Load Face Recognition model (FaceNet)
    fr_model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    for param in fr_model.parameters():
        param.requires_grad = False

    # VGG model for perceptual loss (using first 16 layers)
    vgg_full = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).to(device)
    vgg = nn.Sequential(*list(vgg_full.features.children())[:16]).eval()
    for param in vgg.parameters():
        param.requires_grad = False

    # Loss functions and optimizer
    pixel_loss_fn = nn.L1Loss().to(device)
    optimizer = optim.Adam(model_gen.parameters(), lr=1e-5)

    # Loss weight coefficients
    lambda_pixel = 1.0
    lambda_perceptual = 0.1
    lambda_adv = 0.01

    # Initialize GradScaler for mixed precision training
    scaler = GradScaler()

    # ----------------------
    # Training
    # ----------------------
    num_epochs = 100
    print("Starting training...")
    for epoch in range(1, num_epochs + 1):
        train_model(
            model_gen, train_loader, optimizer, pixel_loss_fn, vgg, fr_model, device, epoch,
            lambda_pixel, lambda_perceptual, lambda_adv, scaler
        )

    # Save the trained model
    torch.save(model_gen.state_dict(), "output/swinir_fr_model.pth")
    print("Training completed and model saved.")

    # ----------------------
    # Evaluation
    # ----------------------
    print("Evaluating model...")
    metrics = evaluate_model(model_gen, test_loader, pixel_loss_fn, device)
    fooling_rate = evaluate_fooling_rate(model_gen, fr_model, test_loader, device)

    print(f"Evaluation Metrics:\n"
          f"  L1 Loss: {metrics['L1']:.4f}\n"
          f"  PSNR: {metrics['PSNR']:.2f} dB\n"
          f"  SSIM: {metrics['SSIM']:.4f}\n"
          f"  Fooling Rate: {fooling_rate:.4f}")

    # Save metrics to a text file
    with open(os.path.join("output", "metrics.txt"), "w") as f:
        f.write(f"L1: {metrics['L1']:.4f}\n")
        f.write(f"PSNR: {metrics['PSNR']:.2f}\n")
        f.write(f"SSIM: {metrics['SSIM']:.4f}\n")
        f.write(f"Fooling Rate: {fooling_rate:.4f}\n")
    print("Evaluation metrics saved.")


if __name__ == "__main__":
    main()
