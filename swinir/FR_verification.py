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
    to_pil = transforms.ToPILImage()
    return to_pil(sr_tensor.cpu())

def load_image(image_input, transform: transforms.Compose) -> torch.Tensor:
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

def process_ground_truth(identity_dir: str, transform: transforms.Compose, model: InceptionResnetV1, device: torch.device) -> torch.Tensor:
    embeddings = []
    for filename in sorted(os.listdir(identity_dir))[:10]:
        image_path = os.path.join(identity_dir, filename)
        image_tensor = load_image(image_path, transform)
        embedding = get_embedding(image_tensor, model, device)
        embeddings.append(embedding)
    return torch.mean(torch.stack(embeddings), dim=0)

def get_reference_embedding(identity_dir: str, transform: transforms.Compose, model: InceptionResnetV1, device: torch.device, save_dir: str) -> torch.Tensor:
    identity_name = os.path.basename(identity_dir)
    embedding_file = os.path.join(save_dir, f"{identity_name}.pt")
    if os.path.exists(embedding_file):
        ref_embedding = torch.load(embedding_file)
    else:
        ref_embedding = process_ground_truth(identity_dir, transform, model, device)
        os.makedirs(save_dir, exist_ok=True)
        torch.save(ref_embedding, embedding_file)
    return ref_embedding

def load_all_reference_embeddings(ground_truth_base: str, transform: transforms.Compose, model: InceptionResnetV1, device: torch.device, save_dir: str) -> dict:
    reference_embeddings = {}
    for identity in os.listdir(ground_truth_base):
        identity_dir = os.path.join(ground_truth_base, identity)
        if not os.path.isdir(identity_dir): continue
        ref_embedding = get_reference_embedding(identity_dir, transform, model, device, save_dir)
        reference_embeddings[identity] = ref_embedding
    return reference_embeddings

def match_query_image_single(query_image_path: str, sr_model: nn.Module, reference_embeddings: dict,
                             fr_transform: transforms.Compose, fr_model: InceptionResnetV1,
                             device: torch.device, threshold: float) -> dict:
    lr_image = Image.open(query_image_path).convert('RGB')
    sr_image = super_resolve_image(lr_image, sr_model, device)
    image_tensor = load_image(sr_image, fr_transform)
    query_embedding = get_embedding(image_tensor, fr_model, device)
    best_match, best_similarity = None, -1.0
    for identity, ref_embedding in reference_embeddings.items():
        similarity = compute_cosine_similarity(query_embedding, ref_embedding)
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = identity
    return {
        "query_image": os.path.basename(query_image_path),
        "best_match": best_match if best_similarity >= threshold else None,
        "best_similarity": best_similarity
    }

def main():
    parser = argparse.ArgumentParser(description="FR Verification with SR-enhanced Queries")
    parser.add_argument('--query_dir', type=str, required=True, help="Directory with LR query images")
    parser.add_argument('--ground_truth_base', type=str, default='./fr_verification/selected_identities', help="Directory with identity subdirectories")
    parser.add_argument('--save_embeddings_dir', type=str, default='./fr_verification/ground_truth_embeddings', help="Directory to save ground truth embeddings")
    parser.add_argument('--sr_model_path', type=str, required=True, help="Path to SR model checkpoint")
    parser.add_argument('--output_dir', type=str, default='./fr_verification/results', help="Directory to save results summary")
    parser.add_argument('--threshold', type=float, default=0.8, help="Cosine similarity threshold")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fr_transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])
    fr_model = InceptionResnetV1(pretrained='vggface2').to(device)
    sr_model = load_sr_model(args.sr_model_path, device)
    reference_embeddings = load_all_reference_embeddings(args.ground_truth_base, fr_transform, fr_model, device, args.save_embeddings_dir)
    
    query_images = [os.path.join(args.query_dir, f) for f in os.listdir(args.query_dir)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    results = []
    for query_image in tqdm(query_images, desc="Processing queries"):
        result = match_query_image_single(query_image, sr_model, reference_embeddings, fr_transform, fr_model, device, args.threshold)
        results.append(result)
    
    os.makedirs(args.output_dir, exist_ok=True)
    model_file = os.path.basename(args.sr_model_path)
    output_file = os.path.join(args.output_dir, f"results_summary_{model_file}.json")
    summary = {"model_used": model_file, "results": results}
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"Summary saved to: {output_file}")

if __name__ == '__main__':
    main()
