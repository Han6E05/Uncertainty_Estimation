import cv2
import os

# Input and output folders
input_folder = "/media/public_data/Projects/extern/Beenen/WP4/Tianhan/dataset/images/test"
output_folder = "/media/public_data/Projects/extern/Beenen/WP4/Tianhan/dataset/images/test2"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop through all files in the input folder
for filename in os.listdir(input_folder):
    file_path = os.path.join(input_folder, filename)

    # Check if it's an image
    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
        img = cv2.imread(file_path)

        if img is None:
            print(f"Skipping {filename}, unable to read.")
            continue

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Save to output folder
        save_path = os.path.join(output_folder, filename)
        cv2.imwrite(save_path, gray)
        print(f"Converted and saved: {save_path}")
