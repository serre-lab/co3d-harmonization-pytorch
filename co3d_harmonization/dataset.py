import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as tvF

# Define data transformations for augmentation and normalization.
data_transforms = {
    'aug': transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=(-15, 15)),
    ]),
    'norm': transforms.Compose([
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),   
}

# -------------------- ClickMe Dataset --------------------
class ClickMe(Dataset):
    """
    A custom Dataset class for the ClickMe dataset.

    Loads image, heatmap, and label data from file paths.
    """
    def __init__(self, image_folder, label_to_category_map, is_training=True):
        """
        Initialize the ClickMe dataset.

        Args:
            image_folder (str): Folder path for images.
            label_to_category_map (dict): Mapping from label names to category indices.
            is_training (bool): Flag to indicate training or validation mode.
        """
        super().__init__()
        self.image_folder = image_folder
        self.label_to_category_map = label_to_category_map
        self.is_training = is_training
        self.data = []
        self.data_dictionary = {}

        if is_training:
            # Setup paths and variables for training images
            image_path = "../CO3D_ClickMe_Training2/"
            co3d_clickme = pd.read_csv("data/CO3D_ClickMe_Training.csv")
            output_dir = "assets"
            image_output_dir = "clickme_test_images"
            image_shape = [256, 256]
            exponential_decay = False

            category_index = 0
            # os.makedirs(image_output_dir, exist_ok=True)
            # os.makedirs(output_dir, exist_ok=True)

            # Process images WITHOUT ClickMaps.
            text_file = "data_lists/co3d_train.txt"
            root_dir = "/oscar/data/tserre/Shared/"
            with open(text_file, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    parts = line.split('/')
                    label = parts[1]
                    path = '/'.join(parts[0:3]) + '/' + parts[3]
                    path = path.split()[1]
                    files = parts[4].split()
                    if label not in self.label_to_category_map.keys():
                        self.label_to_category_map[label] = category_index
                        category_index += 1
                    # Use a subset of files (every 9th image from first 50 images)
                    for i in range(0, 50, 9):
                        full_path = os.path.join(root_dir, path, files[i].strip())
                        rgb_image = Image.open(full_path).convert("RGB")
                        image_name = "_".join(full_path.split("/")[-4:])
                        label = full_path.split("/")[-4]
                        self.data.append({
                            'image': rgb_image,
                            'heatmap': torch.tensor(np.zeros((256, 256))),
                            'category_label': self.label_to_category_map[label],
                            'has_heatmap': False,
                        })
            print("Done processing training images WITHOUT ClickMaps. There are", len(self.data), "images.")

            # Process images WITH ClickMaps.
            data = np.load("co3d_train_processed.npz", allow_pickle=True)
            heatmap_count = 0

            for file in data.files:
                image = data[file][None][0]['image']
                heatmap = data[file][None][0]['heatmap']
                label = file.split("/")[0]
                self.data.append({
                    "image": image, 
                    "heatmap": torch.tensor(heatmap),
                    "category_label": self.label_to_category_map[label],
                    "has_heatmap": True,
                })
                heatmap_count += 1
            print("Done processing training images WITH ClickMaps. There are", heatmap_count, "heatmaps.")

        else:
            # Validation mode: process images with ClickMaps.
            data = np.load("co3d_val_processed.npz", allow_pickle=True)
            heatmap_count = 0

            for file in data.files:
                image = data[file][None][0]['image']
                heatmap = data[file][None][0]['heatmap']
                label = file.split("/")[0]
                self.data.append({
                    "image": image,
                    "heatmap": torch.tensor(heatmap),
                    "category_label": self.label_to_category_map[label],
                    "has_heatmap": True
                })
                heatmap_count += 1
            print("Done processing validation images WITH ClickMaps. There are", heatmap_count, "heatmaps.")
                
    def __getitem__(self, index):
        """
        Retrieve the item at the given index.

        Applies center cropping, preprocessing, and augmentation if in training mode.

        Args:
            index (int): Index of the desired sample.

        Returns:
            tuple: (processed image, processed heatmap, label, has_heatmap flag)
        """
        center_crop = [224, 224]
        img = self.data[index]['image']
        hmp = self.data[index]['heatmap']
        label = self.data[index]['category_label']
        has_heatmap = self.data[index]['has_heatmap']
        
        # Apply center crop and preprocessing to the image.
        img = tvF.center_crop(img, center_crop)
        img = self._preprocess_image(img)
        # Apply center crop and preprocessing to the heatmap.
        hmp = tvF.center_crop(hmp, center_crop)
        hmp = self._preprocess_heatmap(hmp)
        label = self._preprocess_label(label)
        # Apply augmentation if in training mode.
        if self.is_training:
            img, hmp = self._apply_augmentation(img, hmp)
        # Normalize the image.
        img = data_transforms['norm'](img)
        return img, hmp, label, has_heatmap

    def __len__(self):
        """
        Returns the total number of samples.
        """
        return len(self.data)

    def _preprocess_image(self, img):
        """
        Convert a PIL image to a normalized torch tensor.

        Args:
            img (PIL.Image or array): Input image.

        Returns:
            torch.Tensor: Image tensor normalized to [0,1] and with shape [C, H, W].
        """
        img_array = np.array(img)
        img_tensor = torch.from_numpy(img_array)
        # Rearrange dimensions to [C, H, W]
        img_tensor = img_tensor.permute(2, 0, 1)
        return img_tensor.to(torch.float32) / 255.0

    def _preprocess_heatmap(self, hmp):
        """
        Preprocess the heatmap: convert to tensor, and adjust channel dimensions.
        Note: The heatmap is not divided by 255 to preserve its original range.

        Args:
            hmp (PIL.Image, np.array, or torch.Tensor): Input heatmap.

        Returns:
            torch.Tensor: Preprocessed heatmap with shape [1, H, W].
        """
        if not torch.is_tensor(hmp):
            hmp = torch.tensor(np.array(hmp))
        hmp = hmp.to(torch.float32)
        # If heatmap is 2D, add a channel dimension.
        if hmp.ndim == 2:
            hmp = hmp.unsqueeze(0)
        # If heatmap has multiple channels collapse by averaging.
        elif hmp.ndim == 3 and hmp.shape[0] > 1:
            hmp = hmp.mean(dim=0, keepdim=True)
        return hmp

    def _preprocess_label(self, label):
        """
        Preprocess the label by converting it to a torch tensor.

        Args:
            label (int): The label value.

        Returns:
            torch.Tensor: Label tensor of type int64.
        """
        label = torch.tensor(label)
        return torch.squeeze(label.to(torch.int64))

    def _apply_augmentation(self, img, hmp):
        """
        Apply augmentation to both the image and heatmap simultaneously.

        Args:
            img (torch.Tensor): Image tensor with shape [3, H, W].
            hmp (torch.Tensor): Heatmap tensor with shape [1, H, W].

        Returns:
            tuple: Augmented (img, hmp) tensors.
        """
        # Stack image and heatmap to apply same augmentations.
        stacked_img = torch.cat((img, hmp), dim=0)  # shape: [4, H, W]
        stacked_img = data_transforms['aug'](stacked_img)
        # Separate image and heatmap.
        return stacked_img[:3, :, :], stacked_img[3:, :, :]
    
def collate_fn(batch):
    """
    Custom collate function for the DataLoader.

    Args:
        batch (list): List of tuples (image, heatmap, label, image_names).

    Returns:
        tuple: (images, heatmaps, labels, image_names) stacked appropriately.
    """
    images, heatmaps, labels, image_names = zip(*batch)
    images = torch.stack(images, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    heatmaps = torch.stack(heatmaps, dim=0)
    return images, heatmaps, labels, image_names