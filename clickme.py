import re
import os
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr
import torch
import json
from utils import gaussian_kernel, gaussian_blur, create_clickmap
from matplotlib import pyplot as plt
import torch.nn.functional as F


def process_clickmaps(clickmap_csv, is_training=True):
    clickmaps = {}
    num_maps = []
    processed_maps = {}
    n_empty = 0
    for index, row in clickmap_csv.iterrows():
        if is_training:
            # Skipping CO3D Validation Images and Imagenet Images
            if "CO3D_ClickMe2" in row['image_path'] or "imagenet" in row['image_path']:
                continue
            image_file_name = row['image_path'].replace("CO3D_ClickMe_Training2/", "")
        else:
            image_file_name = row['image_path'].replace("CO3D_ClickMe2/", "")
        if image_file_name not in clickmaps.keys():
            clickmaps[image_file_name] = [row["clicks"]]
        else:
            clickmaps[image_file_name].append(row["clicks"])
    
    for image, maps in clickmaps.items():
        n_maps = 0
        for clickmap in maps:
            if len(clickmap) == 2:
                n_empty += 1
                continue
            n_maps += 1
            clean_string = re.sub(r'[{}"]', '', clickmap)
            # Split the string by commas to separate the tuple strings
            tuple_strings = clean_string.split(', ')
            # Zero indexing here because tuple_strings is a list with a single string
            data_list = tuple_strings[0].strip("()").split("),(")
            tuples_list = [tuple(map(int, pair.split(','))) for pair in data_list]

            if image not in processed_maps.keys():
                processed_maps[image] = []
            
            processed_maps[image].append(tuples_list)
        num_maps.append(n_maps)
    return processed_maps, num_maps


def make_heatmap(image_path, point_lists, gaussian_kernel, image_shape, exponential_decay):
    image = Image.open(image_path)
    #PIL image to numpy
    image = np.array(image)
    heatmap = create_clickmap(point_lists, image_shape, exponential_decay=exponential_decay)
    
    # Blur the mask to create a smooth heatmap
    heatmap = torch.from_numpy(heatmap).float().unsqueeze(0)  # Convert to PyTorch tensor
    heatmap = gaussian_blur(heatmap, gaussian_kernel)

    # Check if any maps are all zeros and remove them
    #import pdb; pdb.set_trace()
    zero_maps = heatmap.sum((1, 2)) == 0
    heatmap = heatmap[~zero_maps].squeeze()
  
    # Normalize the heatmap
    # heatmap_normalized = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    heatmap_normalized = heatmap / heatmap.sum()

    # Convert to numpy
    heatmap = heatmap.numpy()  # Convert back to NumPy array         

    # heatmap_normalized = cv2.normalize(heatmap, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    # image_name = "_".join(image_path.split('/')[-2:])
    return image_path, image, heatmap_normalized
