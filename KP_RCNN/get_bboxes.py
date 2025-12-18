import os

# Path to your folder
folder = "dataset/images/test"

# Loop through all files in the folder
for filename in os.listdir(folder):
    # Split name and extension
    name, ext = os.path.splitext(filename)

    # Only process .png files with numeric names
    if ext.lower() == ".png" and name.isdigit():
        old_path = os.path.join(folder, filename)
        # Pad to 6 digits
        new_name = name.zfill(6) + ext
        new_path = os.path.join(folder, new_name)
        os.rename(old_path, new_path)
        print(f"Renamed {filename} -> {new_name}")
