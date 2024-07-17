import os
import cv2
import numpy as np
from skimage import measure
import shutil
from skimage.metrics import structural_similarity as ssim_func

# Set the folder path
folder_path = 'Challenge_Data/Assets'

# Create a new folder for the preview data
preview_data_folder = 'preview_data'
os.makedirs(preview_data_folder, exist_ok=True)

# Iterate through all folders in the data folder
for folder in os.listdir(folder_path):
    folder_path_inner = os.path.join(folder_path, folder)
    if os.path.isdir(folder_path_inner):
        preview_path = os.path.join(folder_path_inner, '_preview.png')

        # Load the _preview image
        preview_img = cv2.imread(preview_path)

        # Get a list of all assets in the folder
        assets = [f for f in os.listdir(folder_path_inner) if os.path.isfile(os.path.join(folder_path_inner, f)) and f != '_preview.png']

        # Initialize an empty list to store the matched assets
        matched_assets = []

        # Loop through each asset in the folder
        for asset in assets:
            asset_path = os.path.join(folder_path_inner, asset)
            asset_img = cv2.imread(asset_path)

            if asset_img is not None:
                # Resize the asset image to the same size as the _preview image
                if preview_img.shape[1] > 0 and preview_img.shape[0] > 0:
                    asset_img = cv2.resize(asset_img, (preview_img.shape[1], preview_img.shape[0]))

                    # Convert both images to grayscale
                    preview_gray = cv2.cvtColor(preview_img, cv2.COLOR_BGR2GRAY)
                    asset_gray = cv2.cvtColor(asset_img, cv2.COLOR_BGR2GRAY)

                    # Compute the Structural Similarity Index (SSIM) between the two images
                    data_range = preview_gray.max() - preview_gray.min()
                    ssim_value = ssim_func(preview_gray, asset_gray, data_range=data_range)

                    if ssim_value > 0.3:
                        matched_assets.append(asset)

        # Create a new folder for the matched assets
        destination_folder = os.path.join(preview_data_folder, folder)
        os.makedirs(destination_folder, exist_ok=True)

        # Copy the matched assets to the new folder
        for asset in matched_assets:
            shutil.copy(os.path.join(folder_path_inner, asset), destination_folder)

        # Copy the _preview.png file to the new folder
        shutil.copy(preview_path, destination_folder)