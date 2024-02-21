import os
from PIL import Image
import shutil

# Function to split image vertically and save left and right halves
def split_and_save(image_path, output_left_path, output_right_path):
    image = Image.open(image_path)
    width, height = image.size
    # Calculate the midpoint
    midpoint = width // 2
    # Split the image
    left_half = image.crop((0, 0, midpoint, height))
    right_half = image.crop((midpoint, 0, width, height))
    # Save left and right halves
    left_half.save(output_left_path)
    right_half.save(output_right_path)

# Directory containing the PNG files
input_directory = r"D:\Unity\Final\Extracted_Frames\Movie_X_45_Y_180"
# Output directories for left and right halves
output_left_directory = r"D:\Unity\Final\Extracted_Frames\RGB_image\Movie_X_45_Y_180"
output_right_directory = r"D:\Unity\Final\Extracted_Frames\Binary_image\Movie_X_45_Y_180"

# Ensure output directories exist, create them if not
os.makedirs(output_left_directory, exist_ok=True)
os.makedirs(output_right_directory, exist_ok=True)

# Iterate over PNG files in the input directory
for filename in os.listdir(input_directory):
    if filename.endswith(".png"):
        # Input image path
        input_image_path = os.path.join(input_directory, filename)
        # Output paths for left and right halves
        output_left_path = os.path.join(output_left_directory, filename)
        output_right_path = os.path.join(output_right_directory, filename)
        # Split and save the image
        split_and_save(input_image_path, output_left_path, output_right_path)
        print(f"Processed: {filename}")

print("All PNG files processed.")

# Delete input directory
shutil.rmtree(input_directory)
print("Input directory deleted.")
