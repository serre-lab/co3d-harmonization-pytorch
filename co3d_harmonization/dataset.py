import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import json
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as tvF

from .config import IMAGE_ITERATOR

CLICKME_DATA_ROOT = "/files22_lrsresearch/CLPS_Serre_Lab/projects/prj_video_imagenet/human_clickme_data_processing/assets/"

# Define data transformations for augmentation and normalization.
data_transforms = {
    'aug': transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
    ]),
    'norm': transforms.Compose([
        transforms.Normalize([0.5000, 0.5000, 0.5000], [0.5000, 0.5000, 0.5000])
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

        # Load the median clicks for each image
        with open('data/co3d_train_medians.json') as json_file:
            co3d_train_medians = json.load(json_file)
        with open('data/co3d_val_medians.json') as json_file:
            co3d_val_medians = json.load(json_file)

        root_dir = "/oscar/data/tserre/Shared/"

        # Process Training Data
        if is_training:
            print("Processing data...")
            co3d_path = "co3d_train"
            # Setup a dictionary to map labels to category indices.
            category_index = 0

            # 1. Process TRAINING images WITHOUT ClickMaps
            text_file = "data_lists/co3d_train.txt" # Text file with all the training images
            with open(text_file, 'r') as file:
                
                lines = file.readlines()

                for line in lines:
                    # Reformat the line to get the image path and label
                    parts = line.split('/')
                    label = parts[1]
                    path = '/'.join(parts[0:3]) + '/' + parts[3]
                    path = path.split()[1]
                    files = parts[4].split()

                    # Crate a new label-category mapping if the label is not in the dictionary
                    if label not in self.label_to_category_map.keys():
                        self.label_to_category_map[label] = category_index
                        category_index += 1

                    # Use a subset of files (every {IMAGE_ITERATOR}th image from first 50 images)
                    for i in range(0, 50, IMAGE_ITERATOR):

                        full_path = os.path.join(root_dir, path, files[i].strip())
                        rgb_image = Image.open(full_path).convert("RGB")
                        image_name = "_".join(full_path.split("/")[-4:])
                        label = full_path.split("/")[-4]
                
                        # Add it to the data list
                        self.data.append({
                            'image': rgb_image,
                            'heatmap': torch.from_numpy(np.zeros((256, 256))),
                            'category_label': self.label_to_category_map[label],
                            'has_heatmap': False,
                            'top_k': 0,
                        })

            print(f"1/4: Done processing training images WITHOUT ClickMaps. There are {len(self.data)} images.")


            # 2. Process TRAINING images WITH ClickMaps.
            data = os.listdir(os.path.join(CLICKME_DATA_ROOT, co3d_path))

            clickmap_count = 0 # Counter to keep track of the number of clickmaps processed

            for file in data:
            
                # Load the Image file and make it a numpy array
                image_name, object_class = self._process_file_name(file)
                image_data = Image.open(os.path.join(root_dir, image_name)).convert("RGB")

                # Get the key for the median clickmap
                image_median_click_key = "/".join(image_name.split('/')[1:])
                
                # Load the clickmap and average it across all maps
                clickmap = np.load(os.path.join(CLICKME_DATA_ROOT, co3d_path, file))
                clickmap = torch.tensor(clickmap).mean(dim=0)

                if object_class not in self.label_to_category_map.keys():
                    self.label_to_category_map[object_class] = category_index
                    category_index += 1

                # Add the image to the data list    
                self.data.append({
                    "image": image_data, 
                    "heatmap": clickmap.clone().detach(),
                    "category_label": self.label_to_category_map[object_class],
                    "has_heatmap": True,
                    'top_k': 0,# co3d_val_medians[image_median_click_key],
                })
                clickmap_count += 1
            print(f"2/4: Done processing training images WITH ClickMaps. There are {clickmap_count} images.")

        # Process Validation data
        else:
            co3d_path = "co3d_val"
            
            text_file = "data_lists/co3d_val.txt" # Text file with all the validation images
            
            # 3. Process VALIDATION images WITHOUT ClickMaps
            with open(text_file, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    # Reformat the line to get the image path and label
                    parts = line.split('/')
                    label = parts[1]
                    path = '/'.join(parts[0:3]) + '/' + parts[3]
                    path = path.split()[1]
                    files = parts[4].split()

                    # Crate a new label-category mapping if the label is not in the dictionary
                    if label not in self.label_to_category_map.keys():
                        self.label_to_category_map[label] = category_index
                        category_index += 1
                    
                    # Use a subset of files (every 9th image from first 50 images)
                    for i in range(0, 50, IMAGE_ITERATOR):

                        full_path = os.path.join(root_dir, path, files[i].strip())
                        rgb_image = Image.open(full_path).convert("RGB")
                        image_name = "_".join(full_path.split("/")[-4:])
                        label = full_path.split("/")[-4]
                        
                        # Add the image to the data list 
                        self.data.append({
                            'image': rgb_image,
                            'heatmap': torch.from_numpy(np.zeros((256, 256))),
                            'category_label': self.label_to_category_map[label],
                            'has_heatmap': False,
                            'top_k': 0,
                        })
            print(f"3/4: Done processing validation images WITHOUT ClickMaps. There are {len(self.data)} images.")

            # 4. Process Validation images WITH ClickMaps.
            data = os.listdir(os.path.join(CLICKME_DATA_ROOT, co3d_path))

            clickmap_count = 0

            for file in data:
            
                # Load the Image file and make it a numpy array
                image_name, object_class = self._process_file_name(file)
                image_data = Image.open(os.path.join(root_dir, image_name)).convert("RGB")
                
                # Load the clickmap and average it across all maps
                clickmap = np.load(os.path.join(CLICKME_DATA_ROOT, co3d_path, file))
                clickmap = torch.tensor(clickmap).mean(dim=0)

                # Get the key for the median clickmap
                image_median_click_key = "/".join(image_name.split('/')[1:])

                if object_class not in self.label_to_category_map.keys():
                    self.label_to_category_map[object_class] = category_index
                    category_index += 1

                # Add the image to the data list    
                self.data.append({
                    "image": image_data, 
                    "heatmap": clickmap.clone().detach(),
                    "category_label": self.label_to_category_map[object_class],
                    "has_heatmap": True,
                    'top_k': 0,# co3d_val_medians[image_median_click_key],
                })
                clickmap_count += 1
            print(f"4/4: Done processing validation images WITH ClickMaps. There are {clickmap_count} images.")

                
    def __getitem__(self, index):
        """
        Retrieve the item at the given index.

        Applies center cropping, preprocessing, and augmentation if in training mode.

        Args:
            index (int): Index of the desired sample.

        Returns:
            tuple: (processed image, processed heatmap, label, has_heatmap flag)
        """
        # Doing the processing here beacause we need to do it for both image and clickmap
        center_crop = [224, 224]
        img = self.data[index]['image']
        hmp = self.data[index]['heatmap']
        label = self.data[index]['category_label']
        has_heatmap = self.data[index]['has_heatmap']
        top_k = self.data[index]['top_k']
        
        # Apply center crop and preprocessing to the image
        img = tvF.center_crop(img, center_crop)
        img = self._preprocess_image(img)

        # Apply center crop and preprocessing to the heatmap
        hmp = tvF.center_crop(hmp, center_crop)
        hmp = self._preprocess_heatmap(hmp)
        label = self._preprocess_label(label)

        # Apply augmentation if in training mode
        if self.is_training:
            img, hmp = self._apply_augmentation(img, hmp)
            
        # Normalize the image.
        img = data_transforms['norm'](img)
        return img, hmp, label, has_heatmap#, top_k

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
    
    def _process_file_name(self, file):
        """
        Get the file name of the image at the given index.

        Args:
            index (int): Index of the desired sample.

        Returns:
            str: File name of the image.
        """
        file_name = file.split("_")
        object_category = file_name[0]
        sequence_id = file_name[1:4]
        frame_number = file_name[5].split(".")[0]
        image_file_name = "binocular_trajectory/" + object_category + "/" + "_".join(sequence_id) + "/renders/" + frame_number + ".png"
        return image_file_name, object_category

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
