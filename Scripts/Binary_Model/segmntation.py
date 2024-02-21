import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import albumentations as albu
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import cv2
from glob import glob

# Dataset class
class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transforms=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        mask = mask / 255.0  # Assuming your mask is scaled between 0-255
        mask = mask.unsqueeze(0)  # Add channel dimension

        return image, mask

# Transformations
def get_training_augmentation():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        ToTensorV2(),
    ]
    return albu.Compose(train_transform)

# Data paths
train_images_dir = 'D:\\Unity\\Final\\Extracted_Frames\\augmented\\train\\rgb\\'
train_masks_dir = 'D:\\Unity\\Final\\Extracted_Frames\\augmented\\train\\binary\\'
test_images_dir = 'D:\\Unity\\Final\\Extracted_Frames\\augmented\\test\\rgb\\'
test_masks_dir = 'D:\\Unity\\Final\\Extracted_Frames\\augmented\\test\\binary\\'

train_img_paths = sorted(glob(os.path.join(train_images_dir, '*.png')))
train_mask_paths = sorted(glob(os.path.join(train_masks_dir, '*.png')))
test_img_paths = sorted(glob(os.path.join(test_images_dir, '*.png')))
test_mask_paths = sorted(glob(os.path.join(test_masks_dir, '*.png')))

# Create Dataset and DataLoader
train_dataset = SegmentationDataset(train_img_paths, train_mask_paths, transforms=get_training_augmentation())
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Model
model = smp.Unet(
    encoder_name="resnet34", 
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader)}")

# Save the model
torch.save(model.state_dict(), 'segmentation_model.pth')
