import os
import json
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from facenet_pytorch import InceptionResnetV1
from models.network_swinir import SwinIR
from tqdm import tqdm

def load_sr_model(model_path: str, device: torch.device) -> nn.Module:
    model = SwinIR(
        upscale=4, img_size=64, window_size=8, img_range=1.0,
        depths=[6,6,6,6,6,6], embed_dim=180, num_heads=[6,6,6,6,6,6],
        mlp_ratio=2, upsampler="pixelshuffle", resi_connection="1conv",
    )
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model

def super_resolve_image(lr_image: Image.Image, sr_model: nn.Module, device: torch.device) -> Image.Image:
    to_tensor = transforms.ToTensor()
    lr_tensor = to_tensor(lr_image).unsqueeze(0).to(device)
    with torch.no_grad():
        sr_tensor = sr_model(lr_tensor)
    sr_tensor = torch.clamp(sr_tensor.squeeze(0), 0, 1)
    return transforms.ToPILImage()(sr_tensor.cpu())

def load_image(image_input, transform: transforms.Compose) -> torch.Tensor:
    # If image_input is a file path, open it; otherwise, assume it is already a PIL image.
    image = Image.open(image_input).convert('RGB') if isinstance(image_input, str) else image_input
    return transform(image)

def get_embedding(image_tensor: torch.Tensor, model: InceptionResnetV1, device: torch.device) -> torch.Tensor:
    model.eval()
    image_tensor = image_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(image_tensor)
    return embedding.squeeze(0).cpu()

def compute_cosine_similarity(emb1: torch.Tensor, emb2: torch.Tensor) -> float:
    return F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()

def process_identity_attack(identity_dir: str, sr_model: nn.Module, fr_transform: transforms.Compose,
                              fr_model: InceptionResnetV1, device: torch.device, threshold: float,
                              skip_sr: bool=False) -> dict:
    # List all image files.
    all_files = [f for f in os.listdir(identity_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if skip_sr:
        # Use the high resolution images: HR_image_1.jpg, HR_image_2.jpg.
        image_files = sorted([f for f in all_files if f.startswith("HR_")])
    else:
        # Use the low resolution images: image_1.jpg, image_2.jpg.
        image_files = sorted([f for f in all_files if not f.startswith("HR_")])
    
    if len(image_files) < 2:
        return None  # Skip identities with fewer than 2 images.
    
    # Use the first two images.
    image_path1 = os.path.join(identity_dir, image_files[0])
    image_path2 = os.path.join(identity_dir, image_files[1])
    
    image1 = Image.open(image_path1).convert('RGB')
    image2 = Image.open(image_path2).convert('RGB')
    
    # If not skipping SR, run the low-resolution images through the SR model.
    processed_image1 = image1 if skip_sr else super_resolve_image(image1, sr_model, device)
    processed_image2 = image2 if skip_sr else super_resolve_image(image2, sr_model, device)
    
    # Apply the FR preprocessing transform.
    tensor1 = load_image(processed_image1, fr_transform)
    tensor2 = load_image(processed_image2, fr_transform)
    
    # Get embeddings.
    emb1 = get_embedding(tensor1, fr_model, device)
    emb2 = get_embedding(tensor2, fr_model, device)
    
    # Compute cosine similarity.
    similarity = compute_cosine_similarity(emb1, emb2)
    # If similarity is below the threshold, the FR model deems them different (attack success).
    attack_success = similarity < threshold
    
    return {
        "identity": os.path.basename(identity_dir),
        "image1": image_files[0],
        "image2": image_files[1],
        "similarity": similarity,
        "attack_success": attack_success
    }

def main():
    parser = argparse.ArgumentParser(description="Attack Success Rate Calculation with SR-enhanced Images")
    parser.add_argument('--identities_dir', type=str, required=True,
                        help="Directory with subdirectories for each identity (each containing 4 images: image_1.jpg, image_2.jpg, HR_image_1.jpg, HR_image_2.jpg)")
    parser.add_argument('--sr_model_path', type=str, help="Path to SR model checkpoint")
    parser.add_argument('--save_results_dir', type=str, default='./fr_verification/attack_results',
                        help="Directory to save results")
    parser.add_argument('--threshold', type=float, default=0.8,
                        help="Cosine similarity threshold (default: 0.8)")
    parser.add_argument('--skip_sr', action='store_true',
                        help="Bypass SR step; use the HR images directly")
    args = parser.parse_args()
    
    if not args.skip_sr and not args.sr_model_path:
        parser.error("--sr_model_path is required unless --skip_sr is enabled")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define the FR preprocessing transform.
    fr_transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    fr_model = InceptionResnetV1(pretrained='vggface2').to(device)
    sr_model = load_sr_model(args.sr_model_path, device) if not args.skip_sr else None
    
    # Iterate over identity subdirectories.
    identities = [os.path.join(args.identities_dir, d)
                  for d in os.listdir(args.identities_dir)
                  if os.path.isdir(os.path.join(args.identities_dir, d))]
    
    results = []
    success_count = 0
    processed_count = 0
    
    for identity_dir in tqdm(identities, desc="Processing identities"):
        result = process_identity_attack(identity_dir, sr_model, fr_transform, fr_model, device, args.threshold, skip_sr=args.skip_sr)
        if result is not None:
            results.append(result)
            processed_count += 1
            if result["attack_success"]:
                success_count += 1
    
    attack_success_rate = success_count / processed_count if processed_count > 0 else 0.0
    
    os.makedirs(args.save_results_dir, exist_ok=True)
    model_file = os.path.basename(args.sr_model_path) if args.sr_model_path else "skipped_sr"
    output_file = os.path.join(args.save_results_dir, f"attack_results_{model_file}.json")
    summary = {
        "model_used": model_file,
        "total_identities_processed": processed_count,
        "attack_successes": success_count,
        "attack_success_rate": attack_success_rate,
        "results": results
    }
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"Summary saved to: {output_file}")
    print(f"Attack Success Rate: {attack_success_rate:.2f}")

if __name__ == '__main__':
    main()
