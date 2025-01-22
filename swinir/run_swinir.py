import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Resize
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from models.network_swinir import SwinIR

# ----------------------
# 1. Model Loading
# ----------------------
def load_swinir_model(pretrained_path):
    model = SwinIR(
        upscale=4, img_size=64, window_size=8,
        img_range=1., depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv'
    )
    checkpoint = torch.load(pretrained_path, weights_only=True)
    model.load_state_dict(checkpoint['params'])
    model.eval()
    return model

print("Loading SwinIR model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = load_swinir_model('./model_zoo/swinir/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth').to(device)
# model = load_swinir_model('./model_zoo/swinir/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN-with-dict-keys-params-and-params_ema.pth').to(device)
print("Model loaded successfully!")

# ----------------------
# 2. Dataset Preparation
# ----------------------
class CelebASRDataset(Dataset):
    def __init__(self, lr_dir, hr_dir):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.filenames = [f for f in os.listdir(lr_dir) if f.endswith(('.png', '.jpg'))]
        
        # Verify one sample
        lr_sample = Image.open(os.path.join(lr_dir, self.filenames[0]))
        hr_sample = Image.open(os.path.join(hr_dir, self.filenames[0]))
        assert lr_sample.size == (32, 32), "LR images must be 32x32"
        assert hr_sample.size == (128, 128), "HR images must be 128x128"

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        lr_img = ToTensor()(Image.open(os.path.join(self.lr_dir, self.filenames[idx])))
        hr_img = ToTensor()(Image.open(os.path.join(self.hr_dir, self.filenames[idx])))
        return lr_img, hr_img

# Init datasets
print("Loading CelebA-SR dataset...")

lr_dir = './datasets/celeba_LR_factor_0.25'
hr_dir = './datasets/celeba_HR_resized_128'

image_set = CelebASRDataset(lr_dir, hr_dir)
print(f"Dataset loaded with {len(image_set)} samples!")

# ----------------------
# 3. Metric Calculation
# ----------------------
def evaluate_model(model, dataset, device):
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    metrics = {'L1': 0, 'MSE': 0, 'PSNR': 0, 'SSIM': 0}
    
    with torch.no_grad():
        for lr, hr in tqdm(loader, desc='Evaluating', dynamic_ncols=True):
            lr = lr.to(device)
            hr = hr.to(device)
            
            # Generate super-resolved image
            sr = model(lr)
            
            # Calculate losses
            metrics['L1'] += torch.nn.L1Loss()(sr, hr).item()
            metrics['MSE'] += torch.nn.MSELoss()(sr, hr).item()
            
            # Convert to numpy (HWC format)
            sr_np = sr.squeeze().cpu().numpy().transpose(1, 2, 0)
            hr_np = hr.squeeze().cpu().numpy().transpose(1, 2, 0)
            
            # Calculate image metrics
            metrics['PSNR'] += psnr(hr_np, sr_np, data_range=1.0)
            metrics['SSIM'] += ssim(hr_np, sr_np, data_range=1.0, channel_axis=-1)

    # Average results
    num_samples = len(dataset)
    return {k: v/num_samples for k, v in metrics.items()}

metrics = evaluate_model(model, image_set, device)
print(f"Metrics:")
print(f"L1: {metrics['L1']:.4f}")
print(f"MSE: {metrics['MSE']:.4f}")
print(f"PSNR: {metrics['PSNR']:.2f} db")
print(f"SSIM: {metrics['SSIM']:.4f}")

# Save metrics
with open('metrics.txt', 'w') as f:
    f.write(f"L1: {metrics['L1']:.4f}\n")
    f.write(f"MSE: {metrics['MSE']:.4f}\n")
    f.write(f"PSNR: {metrics['PSNR']:.2f}\n")
    f.write(f"SSIM: {metrics['SSIM']:.4f}\n")