import os
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from facenet_pytorch import InceptionResnetV1
from models.network_swinir import SwinIR

def load_sr_model(model_path: str, device: torch.device) -> nn.Module:
    """
    Load the saved Super Resolution (SR) model.
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
    checkpoint = torch.load(model_path, map_location=device)
    # The checkpoint may contain a "params" key as in training
    state_dict = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model

def super_resolve_image(lr_image: Image.Image, sr_model: nn.Module, device: torch.device) -> Image.Image:
    """
    Converts an LR image to a super-resolved HR image using the SR model.
    """
    # For the SR model, we use a simple ToTensor transformation (values in [0,1])
    to_tensor = transforms.ToTensor()
    lr_tensor = to_tensor(lr_image).unsqueeze(0).to(device)  # shape: (1, C, H, W)
    
    with torch.no_grad():
        sr_tensor = sr_model(lr_tensor)
    # Clamp to valid range and remove batch dimension
    sr_tensor = torch.clamp(sr_tensor.squeeze(0), 0, 1)
    # Convert tensor back to PIL Image
    to_pil = transforms.ToPILImage()
    sr_image = to_pil(sr_tensor.cpu())
    return sr_image

def load_image(image_path: str, transform: transforms.Compose) -> torch.Tensor:
    """
    Loads an image and applies the transformation pipeline.
    """
    return transform(image_path)

def get_embedding(image_tensor: torch.Tensor, model: InceptionResnetV1, device: torch.device) -> torch.Tensor:
    """
    Computes the FaceNet embedding for a given image tensor.
    """
    model.eval()
    image_tensor = image_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(image_tensor)
    return embedding.squeeze(0).cpu()

def compute_cosine_similarity(emb1: torch.Tensor, emb2: torch.Tensor) -> float:
    """
    Returns the cosine similarity between two embedding vectors.
    """
    return F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()

def process_ground_truth(identity_dir: str, transform: transforms.Compose, model: InceptionResnetV1, device: torch.device) -> torch.Tensor:
    """
    Computes the average embedding from the first 10 images of an identity's directory.
    """
    embeddings = []
    for filename in sorted(os.listdir(identity_dir))[:10]:
        image_path = os.path.join(identity_dir, filename)
        image_tensor = load_image(image_path, transform)
        embedding = get_embedding(image_tensor, model, device)
        embeddings.append(embedding)
    return torch.mean(torch.stack(embeddings), dim=0)

def get_reference_embedding(identity_dir: str, transform: transforms.Compose, model: InceptionResnetV1, device: torch.device, save_dir: str) -> torch.Tensor:
    """
    Loads the saved embedding for an identity if available; otherwise computes, saves, and returns it.
    """
    identity_name = os.path.basename(identity_dir)
    embedding_file = os.path.join(save_dir, f"{identity_name}.pt")
    
    if os.path.exists(embedding_file):
        print(f"Loading saved embedding for identity: {identity_name}")
        ref_embedding = torch.load(embedding_file)
    else:
        print(f"Computing embedding for identity: {identity_name}")
        ref_embedding = process_ground_truth(identity_dir, transform, model, device)
        os.makedirs(save_dir, exist_ok=True)
        torch.save(ref_embedding, embedding_file)
    return ref_embedding

def load_all_reference_embeddings(ground_truth_base: str, transform: transforms.Compose, model: InceptionResnetV1, device: torch.device, save_dir: str) -> dict:
    """
    Loads (or computes) reference embeddings for all identities in the ground truth directory.
    Returns a dictionary mapping identity names to their embeddings.
    """
    reference_embeddings = {}
    for identity in os.listdir(ground_truth_base):
        identity_dir = os.path.join(ground_truth_base, identity)
        if not os.path.isdir(identity_dir):
            continue
        ref_embedding = get_reference_embedding(identity_dir, transform, model, device, save_dir)
        reference_embeddings[identity] = ref_embedding
    return reference_embeddings

def match_query_image(query_image_path: str, sr_model: nn.Module, reference_embeddings: dict,
                      fr_transform: transforms.Compose, fr_model: InceptionResnetV1,
                      device: torch.device, threshold: float) -> None:
    """
    Runs the LR query image through the SR model to super-resolve it, then computes the query image's embedding
    and compares it to each reference embedding. Prints the best matching identity if its similarity exceeds the threshold;
    otherwise, prints 'No match'.
    """
    # Load the LR query image
    lr_image = Image.open(query_image_path).convert('RGB')
    # Super resolve the LR image
    sr_image = super_resolve_image(lr_image, sr_model, device)
    
    # Now apply the FR transformation (e.g., resize to 160x160, normalization)
    image_tensor = load_image(sr_image, fr_transform)
    query_embedding = get_embedding(image_tensor, fr_model, device)
    
    best_match = None
    best_similarity = -1.0
    for identity, ref_embedding in reference_embeddings.items():
        similarity = compute_cosine_similarity(query_embedding, ref_embedding)
        print(f"Similarity with {identity}: {similarity:.4f}")
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = identity
    if best_similarity >= threshold:
        print(f"\nBest match: {best_match} with similarity {best_similarity:.4f}")
    else:
        print(f"\nNo match. Best similarity was {best_similarity:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Face Recognition Verification with SR-enhanced Query")
    parser.add_argument('--query', type=str, required=True, help="Path to the LR query image")
    parser.add_argument('--ground_truth_base', type=str, default='./ground_truth',
                        help="Directory with subdirectories for each identity (each containing at least 10 HR images)")
    parser.add_argument('--save_embeddings_dir', type=str, default='./ground_truth_embeddings',
                        help="Directory to save computed ground truth embeddings")
    parser.add_argument('--sr_model_path', type=str, required=True, help="Path to the saved SR model checkpoint")
    parser.add_argument('--threshold', type=float, default=0.8, help="Cosine similarity threshold for a valid match")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Transformation for the Face Recognition model (e.g., InceptionResnetV1)
    fr_transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])
    
    # Load the Face Recognition model using VGGFace2 pretrained weights
    fr_model = InceptionResnetV1(pretrained='vggface2').to(device)
    
    # Load the SR model and set it to evaluation mode
    sr_model = load_sr_model(args.sr_model_path, device)
    
    # Load or compute reference embeddings for all identities
    reference_embeddings = load_all_reference_embeddings(
        args.ground_truth_base, fr_transform, fr_model, device, args.save_embeddings_dir
    )
    
    # Perform matching for the provided query image
    match_query_image(args.query, sr_model, reference_embeddings, fr_transform, fr_model, device, args.threshold)

if __name__ == '__main__':
    main()
