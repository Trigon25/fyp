import os
import shutil
import itertools
import torch
from torchvision import transforms
from PIL import Image
from facenet_pytorch import InceptionResnetV1

# ---------------------------
# Utility Functions
# ---------------------------
def load_sr_model(model_path: str, device: torch.device):
    from models.network_swinir import SwinIR
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

def super_resolve_image(lr_image: Image.Image, sr_model, device: torch.device) -> Image.Image:
    to_tensor = transforms.ToTensor()
    lr_tensor = to_tensor(lr_image).unsqueeze(0).to(device)
    with torch.no_grad():
        sr_tensor = sr_model(lr_tensor)
    sr_tensor = torch.clamp(sr_tensor.squeeze(0), 0, 1)
    return transforms.ToPILImage()(sr_tensor.cpu())

def compute_fr_embedding_for_candidate(image_filename, src_dir, lr_dir, skip_sr, sr_model, device, transform, fr_model):
    """
    Given an image filename, load the corresponding HR or LR image (optionally applying SR),
    apply the transform, and compute the FR embedding.
    """
    if skip_sr:
        # Use HR image from src_dir directly.
        image_path = os.path.join(src_dir, image_filename)
        image = Image.open(image_path).convert('RGB')
    else:
        # Use LR image from lr_dir, then apply SR.
        image_path = os.path.join(lr_dir, image_filename)
        image = Image.open(image_path).convert('RGB')
        image = super_resolve_image(image, sr_model, device)
    image_tensor = transform(image)
    with torch.no_grad():
        embedding = fr_model(image_tensor.unsqueeze(0).to(device))
    return embedding.squeeze(0).cpu()

# ---------------------------
# Setup: Load Models and Transforms
# ---------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the FR preprocessing transform.
fr_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load the FR model (InceptionResnetV1 pretrained on VGGFace2).
fr_model = InceptionResnetV1(pretrained='vggface2').to(device)
fr_model.eval()

# Set the mode: if True, skip SR and use HR images directly; otherwise, process LR images with SR.
skip_sr = True  # Change to False if you want to super-resolve the LR images.
if not skip_sr:
    # Path to your SR model checkpoint; update accordingly.
    sr_model_path = 'path_to_sr_model_checkpoint.pth'
    sr_model = load_sr_model(sr_model_path, device)
else:
    sr_model = None

# ---------------------------
# Build the Identity Dictionary
# ---------------------------
with open('identity_CelebA.txt', 'r') as file:
    lines = file.readlines()

identity_dict = {}
for line in lines:
    image, identity = line.strip().split()
    identity_dict.setdefault(identity, []).append(image)

# Filter identities that have at least 4 images.
filtered_identities = {ident: imgs for ident, imgs in identity_dict.items() if len(imgs) >= 4}

# Select the first 1000 identities (using insertion order).
selected_identities = list(filtered_identities.keys())[:1000]

# ---------------------------
# Define Directories for Images and Output
# ---------------------------
src_dir = os.path.join("..", "datasets", "celeba_HR_resized_128")   # High-resolution images directory.
lr_dir = os.path.join("..", "datasets", "celeba_LR_factor_0.25")     # Low-resolution images directory.
parent_dir = os.path.join(os.getcwd(), "positive_pairs")
if os.path.exists(parent_dir):
    shutil.rmtree(parent_dir)
os.makedirs(parent_dir)

# ---------------------------
# Process Each Identity and Select the Best Pair
# ---------------------------
# For each identity, we want to select a pair of images whose FR similarity is not too high (<= 0.97).
# Among acceptable pairs, if any have similarity >= 0.8, we choose the best among them.
# Otherwise, we choose the best pair (with highest similarity) among those below 0.97.
for identity in selected_identities:
    candidate_images = filtered_identities[identity]
    # Only consider images that exist in both HR and LR directories.
    valid_candidates = []
    for img in candidate_images:
        hr_path = os.path.join(src_dir, img)
        lr_path = os.path.join(lr_dir, img)
        if os.path.exists(hr_path) and os.path.exists(lr_path):
            valid_candidates.append(img)
    if len(valid_candidates) < 2:
        print(f"Not enough valid images for identity {identity}")
        continue

    # Compute FR embeddings for each candidate.
    embeddings = {}
    for img in valid_candidates:
        emb = compute_fr_embedding_for_candidate(img, src_dir, lr_dir, skip_sr, sr_model, device, fr_transform, fr_model)
        embeddings[img] = emb

    # Evaluate all unique pairs and keep only those with similarity <= 0.97.
    candidate_pairs = []
    for img1, img2 in itertools.combinations(valid_candidates, 2):
        emb1 = embeddings[img1]
        emb2 = embeddings[img2]
        sim = torch.nn.functional.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
        if sim <= 0.97:
            candidate_pairs.append((img1, img2, sim))

    if not candidate_pairs:
        print(f"No acceptable pair for identity {identity} (all pairs too similar)")
        continue

    # First, try to pick a pair with similarity >= 0.85.
    candidates_above_threshold = [pair for pair in candidate_pairs if pair[2] >= 0.85]
    if candidates_above_threshold:
        # Choose the pair with the highest similarity among those.
        best_pair_tuple = min(candidates_above_threshold, key=lambda x: x[2])
    else:
        # Otherwise, choose the pair with the highest similarity overall among candidates.
        best_pair_tuple = max(candidate_pairs, key=lambda x: x[2])
    best_pair = (best_pair_tuple[0], best_pair_tuple[1])
    best_similarity = best_pair_tuple[2]

    # Log the selected pair and its similarity.
    print(f"Identity: {identity}, Selected pair: {best_pair}, Similarity: {best_similarity:.4f}")

    # Create (or clear) the destination subdirectory for this identity.
    dest_dir = os.path.join(parent_dir, identity)
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    else:
        for filename in os.listdir(dest_dir):
            file_path = os.path.join(dest_dir, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

    # Copy the chosen pair's HR and LR images to the destination directory.
    # HR images are renamed with a "HR_" prefix.
    for img in best_pair:
        hr_src = os.path.join(src_dir, img)
        lr_src = os.path.join(lr_dir, img)
        hr_dest = os.path.join(dest_dir, f"HR_{img}")
        lr_dest = os.path.join(dest_dir, img)
        shutil.copy(hr_src, hr_dest)
        shutil.copy(lr_src, lr_dest)
