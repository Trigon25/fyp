import cv2
import os
import shutil
from tqdm import tqdm

def downscale_image(input_path, output_path, scale_factor=0.5):
    """
    Downscale an image by the given scale factor and save it to the output path.

    Parameters:
    - input_path (str): Path to the input HR image.
    - output_path (str): Path to save the downscaled LR image.
    - scale_factor (float): Scaling factor for both width and height.
    """
    # Read the image
    image = cv2.imread(input_path)
    if image is None:
        print(f"Warning: Unable to read image {input_path}. Skipping.")
        return

    # Get original dimensions
    original_height, original_width = image.shape[:2]

    # Calculate new dimensions
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    # Downscale the image
    downscaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Save the resized_downscaled_image
    cv2.imwrite(output_path, downscaled_image)

def process_dataset(input_dir, output_dir, scale_factor=0.5):
    """
    Process all images in the input directory, downscale then resize them, and save to the output directory.

    Parameters:
    - input_dir (str): Root directory containing HR images organized by identity.
    - output_dir (str): Root directory to save LR images.
    - scale_factor (float): Scaling factor for downscaling.
    """
    if not os.path.exists(input_dir):
        raise ValueError(f"Input directory {input_dir} does not exist.")
    else:
        print(f"Input directory contains {len(os.listdir(input_dir))} item(s).", flush=True)

    # Create the output root directory if it doesn't exist
    if os.path.exists(output_dir):
        # Remove the existing directory and its contents
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Walk through the input directory
    for root, _, files in os.walk(input_dir):
        # Compute the relative path from the input root
        rel_path = os.path.relpath(root, input_dir)
        # Compute the corresponding output directory
        target_dir = os.path.join(output_dir, rel_path)
        # Create the output subdirectory if it doesn't exist
        os.makedirs(target_dir, exist_ok=True)

        # Process each file in the current directory
        for file in tqdm(files, desc=f"Processing - factor: {scale_factor}, rel_dir: {rel_path}"):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                input_path = os.path.join(root, file)
                output_path = os.path.join(target_dir, file)
                downscale_image(input_path, output_path, scale_factor)


if __name__ == "__main__":
    cv2.setNumThreads(cv2.getNumberOfCPUs())

    input_res = 128
    input_folder = f"./celeba_HR_resized_{input_res}"
    
    scale_factors = [0.25] # 2x, 4x, 8x
    
    print(f"Running job from dir {input_folder}, scale factor: {scale_factors}")
    for scale_factor in scale_factors:
        output_folder = f"./celeba_LR_factor_{scale_factor}_threaded"
        process_dataset(input_folder, output_folder, scale_factor)
    
    print("Process complete!", flush=True)
