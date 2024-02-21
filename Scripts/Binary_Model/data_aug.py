import os 
import cv2
import numpy as np
from glob import glob
from albumentations import HorizontalFlip, VerticalFlip, Rotate, Compose

#directory having images
directory = 'D:\\Unity\\Final\\Extracted_Frames\\'

def load_data(path):
    # Subfolders for training data
    train_subfolders = ['Movie_X_0_Y_180', 'Movie_X_45_Y_180', 'Movie_X_90_Y_180', 'Movie_X_340_Y_180']
    # Subfolders for test data
    test_subfolders = ['Movie_X_135_Y_180', 'Movie_X_180_Y_180', 'Movie_X_200_Y_180']

    train_img = []
    train_mask = []
    test_img = []
    test_mask = []

    # Iterate through each subfolder and collect training image and mask paths
    for subfolder in train_subfolders:
        train_img.extend(sorted(glob(os.path.join(path, 'RGB_Image', subfolder, '*.png'))))
        train_mask.extend(sorted(glob(os.path.join(path, 'Binary_images', subfolder, '*.png'))))

    # Iterate through each subfolder and collect test image and mask paths
    for subfolder in test_subfolders:
        test_img.extend(sorted(glob(os.path.join(path, 'RGB_Image', subfolder, '*.png'))))
        test_mask.extend(sorted(glob(os.path.join(path, 'Binary_images', subfolder, '*.png'))))

    return (train_img, train_mask), (test_img, test_mask)


def create_dir(sub_path):
    full_path = os.path.join(directory, sub_path)
    if not os.path.exists(full_path):
        os.makedirs(full_path)

def resize_image(image, max_size=512):
    
    #Resize an image while maintaining aspect ratio.
    
    h, w = image.shape[:2]
    scale = min(max_size / h, max_size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    return cv2.resize(image, (new_w, new_h))

def normalize_image(image):
    #Normalize the image.
    
    image = image.astype(np.float32) / 255.0
    return image * 255.0

def augment_image(image, augmentations):
    #Apply augmentations to the image.
    
    augmented = augmentations(image=image)
    return augmented['image']

def augment_data(images, masks, save_path, augment=True):
    # Define augmentations
    augmentations = Compose([HorizontalFlip(p=0.5), VerticalFlip(p=0.5), Rotate(limit=45, p=0.5)], additional_targets={'mask': 'image'})

    for i, (img_path, mask_path) in enumerate(zip(images, masks)):
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0)  # Read mask in grayscale

        image = resize_image(image)
        mask = resize_image(mask)

        if augment:
            # Apply the same augmentation to both image and mask
            augmented = augmentations(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        image = normalize_image(image)

        # Save the processed images and masks
        cv2.imwrite(os.path.join(save_path, 'rgb', f"image_{i}.png"), image)
        cv2.imwrite(os.path.join(save_path, 'binary', f"mask_{i}.png"), mask)

     
if __name__ == '__main__':
    np.random.seed(42)
    
    #loading the data
    data_path = directory
    (train_img, train_mask), (test_img, test_mask) = load_data(data_path)
    print(f"Train: {len(train_img)} (RGB images) --AND-- {len(train_mask)}(Binary images)")
    print(f"Test: {len(test_img)} (RGB images) --AND-- {len(test_mask)}(Binary images)")
    
    #Create a directory to save the augmented images
    create_dir('augmented/train/rgb/')
    create_dir('augmented/train/binary/')
    create_dir('augmented/test/rgb/')
    create_dir('augmented/test/binary/')
    
    # Augment only training data
    augment_data(train_img, train_mask, os.path.join(directory, 'augmented/train/'), augment=True)

    # Directly copy test data without augmentation
    for i, (img_path, mask_path) in enumerate(zip(test_img, test_mask)):
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0)


        image = resize_image(image)
        mask = resize_image(mask)

        image = normalize_image(image)

        # Save the test images and masks without augmentation
        cv2.imwrite(os.path.join(directory, 'augmented/test/rgb/', f"image_{i}.png"), image)
        cv2.imwrite(os.path.join(directory, 'augmented/test/binary/', f"mask_{i}.png"), mask)