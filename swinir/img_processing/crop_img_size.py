import os
import cv2
from tqdm import tqdm

def center_crop_resize(img, target_size):
    """
    Center-crops the given image to a square based on the smaller side,
    then resizes it to (target_size, target_size).
    """
    h, w = img.shape[:2]
    
    # Find the smaller dimension
    min_dim = min(h, w)
    
    # Calculate the top-left corner of the crop
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    
    # Crop the image to a square
    cropped_img = img[start_y:start_y+min_dim, start_x:start_x+min_dim]
    
    # Resize to target_size x target_size
    resized_img = cv2.resize(cropped_img, (target_size, target_size), interpolation=cv2.INTER_AREA)
    
    return resized_img

def process_celebA_images(input_dir, output_dir, target_size=128):
    """
    Center-crops and resizes CelebA images to a square of `target_size`.
    Saves the processed images in `output_dir`.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    input_dir_len = len(os.listdir(input_dir))
    for file_name in tqdm(os.listdir(input_dir), total=input_dir_len, desc="Processing images"):
        if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(input_dir, file_name)
            img = cv2.imread(img_path)
            if img is not None:
                resized = center_crop_resize(img, target_size)
                
                # Construct output path
                out_path = os.path.join(output_dir, file_name)
                cv2.imwrite(out_path, resized)
            else:
                print(f"Warning: Could not read image {img_path}")

if __name__ == "__main__":
    target_size = 128
    input_folder = "./img_align_celeba"
    output_folder = f"./celeba_resized_{target_size}"
    
    process_celebA_images(input_folder, output_folder, target_size=target_size)
