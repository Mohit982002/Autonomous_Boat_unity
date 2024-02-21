import os
from PIL import Image
import numpy as np
from pathlib import Path

def segment_image_rgba(img_array, target_color, tolerance=60):
    distance = np.sqrt(np.sum((img_array - target_color) ** 2, axis=-1))
    mask = distance < tolerance
    segmented_image_array = np.ones((img_array.shape[0], img_array.shape[1], 3), dtype=np.uint8) * 255
    segmented_image_array[mask] = [0, 0, 0]
    segmented_image = Image.fromarray(segmented_image_array)
    return segmented_image

def segment_images_in_folder(input_folder, output_folder, target_color, tolerance=60):
    # Make sure output directory exists, if not, create it
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Iterate over all files in the input directory
    for file_name in os.listdir(input_folder):
        # Construct the full file path
        file_path = os.path.join(input_folder, file_name)
        # Check if it is a file and has a valid image extension
        if os.path.isfile(file_path) and file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Load the image
            image = Image.open(file_path)
            image = image.convert('RGB')  # Ensure it's in RGB format
            # Convert image to numpy array
            img_array = np.array(image)
            # Segment the image
            segmented_img = segment_image_rgba(img_array, target_color, tolerance)
            # Construct the output file path
            output_path = os.path.join(output_folder, file_name)
            # Save the segmented image
            segmented_img.save(output_path)

# Define the input and output folders
input_folder = 'D:\\Unity\\Final\\Extracted_Frames\\Binary_image\\Movie_X_200_Y_180'
output_folder = 'D:\\Unity\\Final\\Extracted_Frames\\Binary_images\\Movie_X_200_Y_180'

# Define the target color (purple)
target_color = np.array([255, 0, 255])

# Call the function to process all images in the folder
segment_images_in_folder(input_folder, output_folder, target_color)
