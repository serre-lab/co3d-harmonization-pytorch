import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import json
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as tvF

from .utils import get_circle_kernel
from .config import IMAGE_ITERATOR, KERNEL_SIZE, KERNEL_SIZE_SIGMA

CLICKME_DATA_ROOT = "/files22_lrsresearch/CLPS_Serre_Lab/projects/prj_video_imagenet/human_clickme_data_processing/assets/"

# A) For images WITHOUT heatmaps in training mode
image_transforms_train = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(0.75, 1.3333)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=(0.6, 1.4), contrast=(0.6, 1.4), saturation=(0.6, 1.4)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# B) For images WITHOUT heatmaps in validation mode
image_transforms_val = transforms.Compose([
    transforms.Resize(256, interpolation=3),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# C) For images WITH heatmaps => resize,crop,tensor,normalize
image_transforms_with_heatmap = transforms.Compose([
    transforms.Resize(256, interpolation=3),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# D) For heatmaps => resize,crop,tensor, but NO normalization
clickmap_transforms_no_norm = transforms.Compose([
    transforms.Resize(256, interpolation=3),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

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
        self.circle_kernel = get_circle_kernel(KERNEL_SIZE, KERNEL_SIZE_SIGMA)

        # Filter out the co3d_train images that overlap with co3d_val
        training_images = os.listdir(os.path.join(CLICKME_DATA_ROOT, "co3d_train"))
        validation_images = os.listdir(os.path.join(CLICKME_DATA_ROOT, "co3d_val"))
        self.overlapping_images = set(training_images).intersection(set(validation_images))

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
                if file not in self.overlapping_images:
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
        img = self.data[index]['image']
        hmp = self.data[index]['heatmap']
        label = self.data[index]['category_label']
        has_heatmap = self.data[index]['has_heatmap']

        # Convert label to torch long
        label = torch.tensor(label, dtype=torch.long)

        # Ensure image is PIL
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img).astype(np.uint8))

        # Ensure heatmap is a NumPy array before we average across channels
        if torch.is_tensor(hmp):
            hmp_np = hmp.cpu().numpy()
        else:
            hmp_np = np.array(hmp, dtype=np.float32)
        
        if hmp_np.ndim == 3 and hmp_np.shape[0] > 1:
            # average over the channel dimension
            hmp_np = hmp_np.mean(axis=0)

        # Convert the single-channel heatmap to a PIL Image (mode='F' for float)
        hmp_img = Image.fromarray(hmp_np, mode='F')

        # If sample HAS heatmap => apply "image_transforms_with_heatmap" to the image
        # and "clickmap_transforms_no_norm" to the heatmap
        if has_heatmap:
            img = image_transforms_with_heatmap(img)          # Resized, center-cropped, normalized
        else:
            # If sample DOES NOT have heatmap => images go to train or val pipeline
            if self.is_training:
                img = image_transforms_train(img)
            else:
                img = image_transforms_val(img)
        
        hmp = clickmap_transforms_no_norm(hmp_img)        # Resized, center-cropped, NO normalization
            
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
