import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from PIL import Image
import re
import os
import utils
from utils import gaussian_kernel, gaussian_blur
from torchvision.transforms import functional as tvF
from clickme import process_clickmaps, make_heatmap
import matplotlib.pyplot as plt


# Define data transformations
data_transforms = {
    'aug': transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.9, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=(-15, 15)),
    ]),
    'norm': transforms.Compose([
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),   
}

BRUSH_SIZE = 11
BRUSH_SIZE_SIGMA = np.sqrt(BRUSH_SIZE)

class ClickMe(Dataset):
    """
    A custom Dataset class for the ClickMe dataset.

    This dataset loads image, heatmap, and label data from file paths,
    and applies optional data augmentation during training.

    Attributes:
        file_paths (list): List of file paths to the data.
        is_training (bool): Flag to indicate if the dataset is for training.
    """

    def __init__(self, image_folder, csv_path, label_to_category_map, is_training=True):
        """
        Initialize the ClickMe dataset.

        Args:
            file_paths (list): List of file paths to the data.
            is_training (bool, optional): Flag to indicate if the dataset is for training. Defaults to True.
        """
        super().__init__()

        self.image_folder = image_folder
        self.csv_path = csv_path
        self.label_to_category_map = label_to_category_map
        self.is_training = is_training
        self.data = []
        self.data_dictionary = {}

        if is_training:
            image_path = "data/CO3D_ClickMe_Training/"
            co3d_clickme = pd.read_csv("data/CO3D_ClickMe_Training.csv")

            output_dir = "assets"
            image_output_dir = "clickme_test_images"
            img_heatmaps = {}
            image_shape = [256, 256]
            thresh = 50
            exponential_decay = False
            plot_images = False

            category_index = 0

            os.makedirs(image_output_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)

            processed_maps, num_maps = process_clickmaps(co3d_clickme, is_training=is_training)
            gaussian_kernel = utils.gaussian_kernel(size=BRUSH_SIZE, sigma=BRUSH_SIZE_SIGMA)
            for idx, (image, maps) in enumerate(processed_maps.items()):
                image_name, image, heatmap = make_heatmap(os.path.join(image_path, image), maps, gaussian_kernel, image_shape=image_shape, exponential_decay=exponential_decay)
                # print(f"image_name: {image_name}")
                label = image_name.split("/")[2]
                # Create label to category and category to label dictionaries
                if label not in self.label_to_category_map.keys():
                    self.label_to_category_map[label] = category_index
                    category_index += 1
                if image_name is None:
                    continue
                # img_heatmaps[image_name] = {"image":image, "heatmap":heatmap}
                image_name = "_".join(image_name.split("/")[-2:])
                self.data_dictionary[image_name] = {"image":image, "heatmap":heatmap, "category_label":self.label_to_category_map[label]}
                # print("1 image_name: ", image_name)

                if len(label_to_category_map.keys()) == 51:
                    break

            print("Done processing training images WITH ClickMaps.")

            # TODO: Change to Train
            # TODO: Change recursive "projects"
            text_file = "/cifs/data/tserre_lrs/projects/projects/prj_video_imagenet/CausalVisionModeling/data_lists/filtered_binocular_renders_test.txt"
            root_dir = "/cifs/data/tserre_lrs/projects/projects/prj_video_imagenet/PeRFception/data/co3d_v2/"

            with open(text_file, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    parts = line.split('/')
                    label = parts[1]
                    path = '/'.join(parts[0:3]) + '/' + parts[3]
                    files = parts[4].split()

                    for i in range(0,50,9):
                        full_path = os.path.join(root_dir, path, files[i].strip())
                        # print(f"2 full_path: {full_path}")
                        image_name = "_".join(full_path.split("/")[-4:])
                        label = full_path.split("/")[-4]
                        # print(f"2 image_name: {image_name}")

                        if image_name in self.data_dictionary.keys():
                            self.data.append({
                                'image': self.data_dictionary[image_name]['image'],
                                'heatmap': self.data_dictionary[image_name]['heatmap'],
                                'category_label': self.data_dictionary[image_name]['category_label']
                            })
                        else:
                            self.data.append({
                                'image': full_path,
                                'heatmap': np.array([]),
                                'category_label': self.label_to_category_map[label]
                            })
            print("Done processing training images WITHOUT ClickMaps.")

        else:
            image_path = "data/CO3D_ClickMe_Validation/"
            co3d_clickme = pd.read_csv("data/CO3D_ClickMe_Validation.csv")

            output_dir = "assets"
            image_output_dir = "clickme_test_images"
            img_heatmaps = {}
            image_shape = [256, 256]
            thresh = 50
            exponential_decay = False
            plot_images = False

            category_index = 0

            os.makedirs(image_output_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)

            processed_maps, num_maps = process_clickmaps(co3d_clickme, is_training=is_training)
            gaussian_kernel = utils.gaussian_kernel(size=BRUSH_SIZE, sigma=BRUSH_SIZE_SIGMA)
            for idx, (image, maps) in enumerate(processed_maps.items()):
                # print(f"Vimage: {image}")
                full_path = os.path.join(image_path, image)
                image_name, image, heatmap = make_heatmap(os.path.join(image_path, image), maps, gaussian_kernel, image_shape=image_shape, exponential_decay=exponential_decay)
                # print(f"image_name: {image_name}")
                label = image_name.split("/")[2]
                # Create label to category and category to label dictionaries
                if label not in self.label_to_category_map.keys():
                    self.label_to_category_map[label] = category_index
                    category_index += 1
                if image_name is None:
                    continue
                # img_heatmaps[image_name] = {"image":image, "heatmap":heatmap}
                image_name = "_".join(image_name.split("/")[-2:])
                self.data.append({"image": full_path, "heatmap":heatmap, "category_label":self.label_to_category_map[label]})


    def __getitem__(self, index):
        """
        Get an item from the dataset.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            tuple: A tuple containing the processed image, heatmap, and label.
        """
        center_crop=[224, 224]
        img, hmp, label = self.data[index]['image'], self.data[index]['heatmap'], self.data[index]['category_label']

        print("Data Label:", self.data[index]['category_label'])

        # print("GI img: ", img)  
        image_name = img.split("/")[2]
        img = Image.open(img)
        img = tvF.center_crop(img, center_crop)

        # hmp = Image.open(hmp)
        # print(f"img: {img.size} hmp: {hmp.shape} label: {label}")

        img = self._preprocess_image(img)

        if not len(hmp) == 0:
            hmp = tvF.center_crop(hmp, center_crop)
            hmp = self._preprocess_heatmap(hmp)
        label = self._preprocess_label(label)

        if self.is_training:
            img, hmp = self._apply_augmentation(img, hmp)

        img = data_transforms['norm'](img)  # Apply ImageNet mean and std
        return img, hmp, label, image_name

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        return len(self.data)

    def _preprocess_image(self, img):
        """
        Preprocess the image tensor.

        Args:
            img (torch.Tensor): The input image tensor.

        Returns:
            torch.Tensor: The preprocessed image tensor.
        """
        # Convert PIL Image to numpy array
        img_array = np.array(img)
        # Convert numpy array to torch tensor
        img_tensor = torch.from_numpy(img_array)
        # Rearrange dimensions to [C, H, W] format
        img_tensor = img_tensor.permute(2, 0, 1)
        return img_tensor.to(torch.float32) / 255.0 # Normalize to [0, 1]       

    def _preprocess_heatmap(self, hmp):
        """
        Preprocess the heatmap tensor.

        Args:
            hmp (torch.Tensor): The input heatmap tensor.

        Returns:
            torch.Tensor: The preprocessed heatmap tensor.
        """
        return hmp.to(torch.float32) / 255.0

    def _preprocess_label(self, label):
        """
        Preprocess the label tensor.

        Args:
            label (torch.Tensor): The input label tensor.

        Returns:
            torch.Tensor: The preprocessed label tensor.
        """
        label = torch.tensor(label)
        return torch.squeeze(label.to(torch.int64))

    def _apply_augmentation(self, img, hmp):
        """
        Apply data augmentation to the image and heatmap.

        Args:
            img (torch.Tensor): The input image tensor.
            hmp (torch.Tensor): The input heatmap tensor.

        Returns:
            tuple: A tuple containing the augmented image and heatmap tensors.
        """
        # Add an extra dimension to hmp to make it 3D
        if not len(hmp) == 0:
            hmp = hmp.unsqueeze(0)
        
            stacked_img = torch.cat((img, hmp), dim=0)
            stacked_img = data_transforms['aug'](stacked_img)
            return stacked_img[:-1, :, :], stacked_img[-1, :, :]  # Remove the extra dimension from hmp
        else:
            return data_transforms['aug'](img), np.array([])
