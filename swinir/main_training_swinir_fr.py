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
from torchvision.transforms import ToTensor, Resize
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from models.network_swinir import SwinIR
from facenet_pytorch import MTCNN, InceptionResnetV1

# ----------------------
# 0. Setup
# ----------------------

# Create output directory
os.makedirs("output", exist_ok=True)

# Set Input Directories
lr_dir = "./datasets/celeba_LR_factor_0.25"
hr_dir = "./datasets/celeba_HR_resized_128"

# Set random seed
torch.manual_seed(42)
np.random.seed(42)
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load img filenames from txt files for train and test
with open("img_processing/split_ids/identity_train.txt", "r") as f:
    lines = f.read().splitlines()
    train_img_filenames = [line.split()[0].strip() for line in lines]
    train_img_ids = [line.split()[1] for line in lines]

print(f"Number of training identities: {len(set(train_img_ids))}")
print(f"Number of training images: {len(train_img_filenames)}")

with open("img_processing/split_ids/identity_test.txt", "r") as f:
    lines = f.read().splitlines()
    test_img_filenames = [line.split()[0].strip() for line in lines]
    test_img_ids = [line.split()[1] for line in lines]

print(f"Number of test identities: {len(set(test_img_ids))}")
print(f"Number of test images: {len(test_img_filenames)}")

# ----------------------
# 1. Dataset Preparation
# ----------------------
class CelebASRDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, mode="train"):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.filenames = [
            f.strip() for f in os.listdir(lr_dir) if f.endswith((".png", ".jpg"))
        ]

        if mode == "train":
            self.filenames = list(
                set(self.filenames).intersection(set(train_img_filenames))
            )
        elif mode == "test":
            self.filenames = list(
                set(self.filenames).intersection(set(test_img_filenames))
            )
        else:
            raise ValueError("Invalid mode. Choose 'train' or 'test'.")
        
        # Verify one sample
        lr_sample = Image.open(os.path.join(lr_dir, self.filenames[0]))
        hr_sample = Image.open(os.path.join(hr_dir, self.filenames[0]))
        assert lr_sample.size == (32, 32), "LR images must be 32x32"
        assert hr_sample.size == (128, 128), "HR images must be 128x128"

    def __repr__(self):
        return f"Dataset: {len(self)} samples"

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        lr_img = ToTensor()(Image.open(os.path.join(self.lr_dir, self.filenames[idx])))
        hr_img = ToTensor()(Image.open(os.path.join(self.hr_dir, self.filenames[idx])))
        # return lr_img, hr_img, self.filenames[idx]
        return {"lr": lr_img, "hr": hr_img, "filename": self.filenames[idx]}


# Init datasets
print("Loading CelebA-SR dataset...")
train_set = CelebASRDataset(lr_dir, hr_dir, mode="train")
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
print(f"Dataset loaded.")

# ----------------------
# 2. Model Loading
# ----------------------
def load_swinir_model(pretrained_path):
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
    checkpoint = torch.load(pretrained_path, weights_only=True)
    model.load_state_dict(checkpoint["params"], strict=False)
    return model


print("Loading SwinIR model...")
model_gen = load_swinir_model(
    "./model_zoo/swinir/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth"
).to(device)
for name, param in model_gen.named_parameters():
    if "layers.0" in name or "patch_embed" in name:  # Freeze first layer
        param.requires_grad = False
print("Model loaded successfully!")

# Fixed Facial Recognition Model FaceNet
fr_model = InceptionResnetV1(pretrained="vggface2").eval().requires_grad_(False).to(device)

# VGG for perceptual loss
vgg = (
    models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
    .features[:16]
    .eval()
    .requires_grad_(False)
)

# ----------------------
# 3. Training
# ----------------------
# Pixel Loss (L1)
pixel_loss = nn.L1Loss()


# Perceptual Loss (using VGG or FR model's intermediate layers)
def perceptual_loss(sr, hr):
    # Using VGG features
    sr_features = vgg(sr)
    hr_features = vgg(hr.detach())
    return torch.mean(torch.abs(sr_features - hr_features))


# Adversarial Loss (fool FR model)
def adversarial_loss(sr, hr):
    sr_emb = fr_model(sr)  # Get FR embedding for SR image
    hr_emb = fr_model(hr)  # Get FR embedding for HR image (target)
    # Maximize dissimilarity between SR and HR embeddings
    return 1 - torch.nn.functional.cosine_similarity(sr_emb, hr_emb).mean()


# Optimizer
optimizer = optim.Adam(model_gen.parameters(), lr=1e-5)

# Loss weights
lambda_pixel = 1.0
lambda_perceptual = 0.1
lambda_adv = 0.01

print("Starting training...")
# Training
for epoch in range(100):
    with tqdm(train_loader, unit="batch") as tepoch:
        for batch in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            lr = batch['lr']
            hr = batch['hr']
            
            # Forward pass
            sr = model_gen(lr)
            
            # Loss calculations
            loss_pixel = pixel_loss(sr, hr)
            loss_perceptual = perceptual_loss(sr, hr)
            loss_adv = adversarial_loss(sr, hr)
            
            # Total loss
            total_loss = (
                lambda_pixel * loss_pixel +
                lambda_perceptual * loss_perceptual +
                lambda_adv * loss_adv
            )
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            tepoch.set_postfix(loss=total_loss.item())
    
# Save model
torch.save(model_gen.state_dict(), "output/swinir_fr_model.pth")

# ----------------------    
# 4. Evaluation
# ----------------------
# Visual Quality (PSNR/SSIM)
def calculate_psnr(sr, hr):
    mse = torch.mean((sr - hr) ** 2)
    return 10 * torch.log10(1.0 / mse)

# Fooling Rate (FR accuracy drop)
def evaluate_fooling_rate(generator, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            lr = batch['lr']
            hr = batch['hr']
            sr = generator(lr)
            
            # Get FR predictions
            pred_hr = fr_model(hr).argmax(dim=1)
            pred_sr = fr_model(sr).argmax(dim=1)
            
            # Compare predictions
            total += hr.size(0)
            correct += (pred_sr != pred_hr).sum().item()
    
    return correct / total  # Higher = better fooling rate

# Load test dataset
test_set = CelebASRDataset(lr_dir, hr_dir, mode="test")
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)                                

# Evaluate model
model_gen.eval()                        
metrics = {'L1': 0, 'PSNR': 0, 'SSIM': 0}
fooling_rate = evaluate_fooling_rate(model_gen, test_loader)

with torch.no_grad():
    for batch in test_loader:
        lr = batch['lr']
        hr = batch['hr']
        sr = model_gen(lr)
        
        # Calculate metrics
        metrics['L1'] += pixel_loss(sr, hr).item()
        metrics['PSNR'] += calculate_psnr(sr, hr)
        
        # Convert to numpy (HWC format)
        sr_np = sr.squeeze().cpu().numpy().transpose(1, 2, 0)
        hr_np = hr.squeeze().cpu().numpy().transpose(1, 2, 0)
        
        # Calculate image metrics
        metrics['SSIM'] += ssim(hr_np, sr_np, data_range=1.0, multichannel=True)

# Average results
num_samples = len(test_set)
metrics = {k: v/num_samples for k, v in metrics.items()}
print(f"Metrics:")
print(f"L1: {metrics['L1']:.4f}")
print(f"PSNR: {metrics['PSNR']:.2f} db")
print(f"SSIM: {metrics['SSIM']:.4f}")
print(f"Fooling Rate: {fooling_rate:.4f}")

# Save metrics
with open('output/metrics.txt', 'w') as f:
    f.write(f"L1: {metrics['L1']:.4f}\n")
    f.write(f"PSNR: {metrics['PSNR']:.2f}\n")
    f.write(f"SSIM: {metrics['SSIM']:.4f}\n")
    f.write(f"Fooling Rate: {fooling_rate:.4f}\n")
