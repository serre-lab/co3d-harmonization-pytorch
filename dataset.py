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
from torch.utils.data import DataLoader
import random


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

    def __init__(self, image_folder, label_to_category_map, is_training=True):
        """
        Initialize the ClickMe dataset.

        Args:
            file_paths (list): List of file paths to the data.
            is_training (bool, optional): Flag to indicate if the dataset is for training. Defaults to True.
        """
        super().__init__()

        self.image_folder = image_folder
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

            # TODO: Change recursive "projects"
            text_file = "/cifs/data/tserre_lrs/projects/projects/prj_video_imagenet/CausalVisionModeling/data_lists/filtered_binocular_renders_test.txt"
            root_dir = "/cifs/data/tserre_lrs/projects/projects/prj_video_imagenet/PeRFception/data/co3d_v2/"
            # text_file = "/media/data_cifs/projects/prj_video_imagenet/CausalVisionModeling/data_lists/filtered_binocular_renders_test.txt"
            # root_dir = "/media/data_cifs/projects/projects/prj_video_imagenet/PeRFception/data/co3d_v2/"

            with open(text_file, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    parts = line.split('/')
                    label = parts[1]
                    path = '/'.join(parts[0:3]) + '/' + parts[3]
                    files = parts[4].split()
                    # Create label to category and category to label dictionaries
                    if label not in self.label_to_category_map.keys():
                        self.label_to_category_map[label] = category_index
                        category_index += 1

                    for i in range(0,50, 9):
                        full_path = os.path.join(root_dir, path, files[i].strip())
                        # print(f"2 full_path: {full_path}")
                        image_name = "_".join(full_path.split("/")[-4:])
                        label = full_path.split("/")[-4]
                        # print(f"2 image_name: {image_name}")

                        # print("ImageName: ", image_name, list(self.data_dictionary.keys())[0])

                        
                        self.data.append({
                            'image': full_path,
                            'heatmap': torch.from_numpy(np.zeros((256, 256))).float(),
                            'category_label': self.label_to_category_map[label],
                            'has_heatmap': False,
                        })
            print("Done processing training images WITHOUT ClickMaps.")

            # if image_name in self.data_dictionary.keys():
            #     self.data.append({
            #         'image': self.data_dictionary[image_name]['image'],
            #         'heatmap': self.data_dictionary[image_name]['heatmap'],
            #         'category_label': self.data_dictionary[image_name]['category_label']
            #     })

            processed_maps, num_maps = process_clickmaps(co3d_clickme, is_training=is_training)
            gaussian_kernel = utils.gaussian_kernel(size=BRUSH_SIZE, sigma=BRUSH_SIZE_SIGMA)
            heatmap_count = 0
            for idx, (image, maps) in enumerate(processed_maps.items()):
                full_path = os.path.join(image_path, image)
                image_name, image, heatmap = make_heatmap(os.path.join(image_path, image), maps, gaussian_kernel, image_shape=image_shape, exponential_decay=exponential_decay)
                # print(f"image_name: {image_name}")
                label = image_name.split("/")[2]
                if image_name is None:
                    continue
                # img_heatmaps[image_name] = {"image":image, "heatmap":heatmap}
                image_name = "_".join(image_name.split("/")[-2:])
                self.data.append({"image":full_path, "heatmap":heatmap, "category_label":self.label_to_category_map[label], "has_heatmap":True})
                # print("1 image_name: ", image_name)
                heatmap_count += 1

            print("Done processing training images WITH ClickMaps. There are ", heatmap_count, "heatmaps.")
            # else:

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
                self.data.append({"image": full_path, "heatmap":heatmap, "category_label":self.label_to_category_map[label], "has_heatmap":True})


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

        # print("Data Label:", self.data[index]['category_label'])

        # print("GI img: ", img)  
        image_name = img#.split("/")[2]
        img = Image.open(img)
        img = tvF.center_crop(img, center_crop)

        # hmp = Image.open(hmp)
        # print(f"img: {img.size} hmp: {hmp.shape} label: {label}")

        img = self._preprocess_image(img)

        # if not len(hmp) == 0:
        # hmp = Image.fromarray(hmp)
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
        hmp = torch.tensor(hmp)
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
        # if has_heatmap:
        hmp = hmp.unsqueeze(0)
    
        stacked_img = torch.cat((img, hmp), dim=0)
        stacked_img = data_transforms['aug'](stacked_img)
        return stacked_img[:-1, :, :], stacked_img[-1, :, :]  # Remove the extra dimension from hmp
        # else:
        #     return data_transforms['aug'](img), np.array([])
    
def build_co3d_eval_loader(args, transform=None, return_all=False, label_to_index_map=None):
    cifs = "/cifs/data/tserre_lrs/projects/prj_video_imagenet/"
    if not os.path.exists(cifs):
        cifs = "/cifs/data/tserre_lrs/projects/projects/prj_video_imagenet/"
    TRAIN_LIST_PATH = 'data_lists/filtered_binocular_renders_train.txt'
    DATA_ROOT = os.path.join(cifs, 'PeRFception/data/co3d_v2/binocular_trajectory/')
    DATA_PATH = os.path.join(cifs, 'Evaluation/')
    TEST_DATA_PATH = './assets/co3d_clickmaps_normalized.npy'
    TEST_HUMAN_RESULTS = './assets/human_ceiling_results.npz'
    label_to_index_map = label_to_index_map
    
    # transform_train = transforms.Compose([
    #     transforms.RandomResizedCrop(224, interpolation=3),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize(mean=args.mean, std=args.std)])
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    co3d_dataset_test = Co3dTestDataset(numpy_file=TEST_DATA_PATH, human_results_file=TEST_HUMAN_RESULTS, label_to_index=label_to_index_map, transform=transform)
    co3d_dataloader_test = DataLoader(co3d_dataset_test, batch_size=1, num_workers=args.num_workers, shuffle=False)
    
    if return_all:
        co3d_dataset_train = Co3dLpDataset(root=DATA_ROOT,
                train = True,
                transform=transform,
                datapath = os.path.join(DATA_PATH,"filtered_co3d_train.txt"),
                train_start_frame=0,
                train_end_frame=40,
                val_start_frame=41, 
                val_end_frame=49)
        co3d_dataset_val = Co3dLpDataset(root=DATA_ROOT,
                train = False,
                transform=transform,
                datapath = os.path.join(DATA_PATH, "filtered_co3d_test.txt"),
                train_start_frame=0,
                train_end_frame=40,
                val_start_frame=41, 
                val_end_frame=49)
        co3d_dataloader_train = DataLoader(co3d_dataset_train, batch_size=256, shuffle=False, num_workers=args.num_workers)
        co3d_dataloader_val = DataLoader(co3d_dataset_val, batch_size=256, shuffle=False, num_workers=args.num_workers)
        return co3d_dataloader_train, co3d_dataloader_val, co3d_dataloader_test
    return co3d_dataloader_test

from sklearn import preprocessing

class Co3dLpDataset(torch.utils.data.Dataset):
    def __init__(self,
                 root,
                 train=True,
                 datapath="",
                 transform=None,
                 lazy_init=False,
                 reverse_sequence=False,
                 train_start_frame = 0,
                 train_end_frame = 39,
                 val_start_frame = 40,
                 val_end_frame = 49):

        super(Co3dLpDataset, self).__init__()
        self.root = root
        self.train = train
        self.datapath = datapath,
        self.transform = transform
        self.lazy_init = lazy_init
        self.reverse_sequence = reverse_sequence
        self.train_start_frame = train_start_frame
        self.train_end_frame = train_end_frame
        self.val_start_frame = val_start_frame
        self.val_end_frame = val_end_frame

        if not self.lazy_init:
            self.clips = self.make_dataset_samples()
            if len(self.clips) == 0:
                raise(RuntimeError("Found 0 video clips in subfolders of: " + root + "\n"
                                   "Check your data directory (opt.data-dir)."))

        # Preprocessing the label
        all_classes = os.listdir(self.root)
        self.label_preprocessing = preprocessing.LabelEncoder()
        self.label_preprocessing.fit(np.array(all_classes))

    def __getitem__(self, index):
        sample = self.clips[index%len(self.clips)]
        image = self.load_frame(sample)

        # Adding the labels
        label = sample[0].split('/')[0]
        label = self.label_preprocessing.transform([label])

        label = torch.tensor(label)

        if self.transform:
            image = self.transform(image)

        return image, label[0]

    def __len__(self):
        return len(self.clips)

    def load_frame(self, sample):
        fname = os.path.join(self.root, sample[0])
        frames_list = sample[1:]

        if not (os.path.exists(fname)):
            print(f"Frame of {fname} does not exist")
            return []

        # Train on only training frames
        if self.train:
            frames_list = frames_list[self.train_start_frame: self.train_end_frame]

        # Validate on remaining frames
        else:
            frames_list = frames_list[self.val_start_frame: self.val_end_frame]

        selected_random_frame = random.choice(frames_list)

        img = Image.open(os.path.join(fname, selected_random_frame)).convert('RGB')  # Open the image

        return img

    def make_dataset_samples(self):
        self.datapath = self.datapath[0]
        with open(self.datapath, 'r') as fopen:
            lines = fopen.readlines()
        all_seqs = [line.split() for line in lines]
        return all_seqs

class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

class Co3dTestDataset(Dataset):
    def __init__(self, numpy_file, human_results_file, label_to_index, transform=None):
        super(Co3dTestDataset, self).__init__()
        self.image_names = []
        self.image_files = []
        self.heatmaps = []
        self.categories = []
        self.transform = transform
        self.label_to_index = label_to_index
        all_data = np.load(numpy_file, allow_pickle=True)
        human_data = np.load(human_results_file, allow_pickle=True)
        # Weird indexing because I didn't save the dictionary right
        all_data = all_data[None][0]
        filtered_imgs = human_data['final_clickmaps_thresholded'].tolist().keys()
        for image_name in all_data.keys():
            cat = image_name.split('_')[0]
            if image_name.replace(f'{cat}_', f'{cat}/') in filtered_imgs:
                self.image_files.append(all_data[image_name]['image'])
                self.heatmaps.append(all_data[image_name]['heatmap'])
                self.image_names.append(image_name)
                self.categories.append(cat)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        img_label = self.image_names[idx].split('_')[0]
        img_name = self.image_names[idx]
        numeric_label = torch.tensor(self.label_to_index[img_label], dtype=torch.long)
        heatmap = self.heatmaps[idx]
        if self.transform:
            image = self.transform(img_file)
        else:
            image = img_file
        # print("Test:", image.shape)
        cat = self.categories[idx]
        return image, heatmap, numeric_label, img_name, cat