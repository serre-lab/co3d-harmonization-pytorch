import argparse
import random
import os
import numpy as np
import pandas as pd
from PIL import Image
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as tvF
import timm
import ipdb
from scipy.stats import spearmanr
from clickme import process_clickmaps, make_heatmap
# Add wandb logging
import wandb    
import matplotlib.pyplot as plt

# Initialize wandb logging if enabled.
wandb_logging = True
if wandb_logging:
    wandb.init(entity="grassknoted", project="co3d-harmonization")

# Global constants
N_CO3D_CLASSES = 51
BRUSH_SIZE = 11
BRUSH_SIZE_SIGMA = np.sqrt(BRUSH_SIZE)
HUMAN_SPEARMAN_CEILING = 0.4422303328731989

def get_gaussian_kernel(kernel_size=BRUSH_SIZE, sigma=BRUSH_SIZE_SIGMA, channels=1):
    """
    Create a Gaussian kernel for blurring.

    Args:
        kernel_size (int): Size of the kernel.
        sigma (float): Standard deviation of the Gaussian.
        channels (int): Number of channels (kernel is repeated for each channel).

    Returns:
        torch.Tensor: Gaussian kernel of shape [channels, 1, kernel_size, kernel_size].
    """
    ax = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size // 2)
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    return kernel.repeat(channels, 1, 1, 1)

class utils:
    """
    Utility class that provides helper functions.
    """
    @staticmethod
    def gaussian_kernel(size, sigma):
        """
        Get a Gaussian kernel using the global get_gaussian_kernel function.

        Args:
            size (int): Kernel size.
            sigma (float): Standard deviation for the Gaussian.

        Returns:
            torch.Tensor: Gaussian kernel tensor.
        """
        return get_gaussian_kernel(kernel_size=size, sigma=sigma, channels=1)

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
            os.makedirs(image_output_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)

            # Process images WITHOUT ClickMaps.
            text_file = "/files22_lrsresearch/CLPS_Serre_Lab/projects/prj_video_imagenet/CausalVisionModeling/data_lists/filtered_binocular_renders_test.txt"
            root_dir = "/files22_lrsresearch/CLPS_Serre_Lab/projects/prj_video_imagenet/PeRFception/data/co3d_v2/"
            with open(text_file, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    parts = line.split('/')
                    label = parts[1]
                    path = '/'.join(parts[0:3]) + '/' + parts[3]
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

def gaussian_blur(x, kernel_size=BRUSH_SIZE, sigma=BRUSH_SIZE_SIGMA):
    """
    Apply Gaussian blur to the input tensor.

    Args:
        x (torch.Tensor or list): Input tensor of shape [B, C, H, W] or a list of tensors.
        kernel_size (int): Kernel size for the Gaussian blur.
        sigma (float): Standard deviation of the Gaussian.

    Returns:
        torch.Tensor: Blurred tensor.
    """
    if isinstance(x, list):
        x = torch.stack(x, dim=0)
    channels = x.shape[1]
    kernel = get_gaussian_kernel(kernel_size, sigma, channels).to(x.device)
    padding = kernel_size // 2
    return F.conv2d(x, kernel, padding=padding, groups=channels)

def build_gaussian_pyramid(x, levels=5):
    """
    Build a Gaussian pyramid for a given tensor.

    Args:
        x (torch.Tensor): Input tensor with shape [B, C, H, W].
        levels (int): Number of pyramid levels.

    Returns:
        list: A list of tensors forming the Gaussian pyramid.
    """
    pyramid = []
    current = x
    for _ in range(1, levels):
        blurred = gaussian_blur(current, kernel_size=BRUSH_SIZE, sigma=BRUSH_SIZE_SIGMA)
        downsampled = F.avg_pool2d(blurred, kernel_size=2, stride=2)
        pyramid.append(blurred)
        current = downsampled
    return pyramid

def compute_spearman_correlation(saliency, clickmap):
    """
    Compute Spearman's rank correlation between two maps.

    Args:
        saliency (torch.Tensor): Saliency map tensor.
        clickmap (torch.Tensor): Clickmap tensor.

    Returns:
        float: Spearman correlation coefficient.
    """
    saliency_np = saliency.view(-1).detach().cpu().numpy()
    clickmap_np = clickmap.view(-1).detach().cpu().numpy()
    correlation, _ = spearmanr(saliency_np, clickmap_np)
    return correlation

def MSE(x, y):
    """
    Compute the Mean Squared Error (MSE) between two tensors.

    Args:
        x (torch.Tensor): First tensor.
        y (torch.Tensor): Second tensor.

    Returns:
        torch.Tensor: Tensor of MSE values computed over the flattened dimensions.
    """
    x = x.reshape(len(x), -1)
    y = y.reshape(len(y), -1)
    return torch.square(x - y).mean(-1)

def CE(q, k):
    """
    Compute a cross entropy loss between two tensors.

    Args:
        q (torch.Tensor): Input tensor.
        k (torch.Tensor): Target tensor.

    Returns:
        torch.Tensor: Cross entropy loss per sample.
    """
    batch_dimension = q.shape[0]
    q_flat = q.view(batch_dimension, -1)
    k_flat = k.view(batch_dimension, -1)
    return F.cross_entropy(q_flat, k_flat.argmax(dim=-1), reduction='none')

def cosine(x, y):
    """
    Compute the cosine distance between two tensors.

    Args:
        x (torch.Tensor): First tensor.
        y (torch.Tensor): Second tensor.

    Returns:
        torch.Tensor: Tensor of cosine distances computed over the flattened dimensions per sample.
    """
    # Flatten the tensors to shape [B, D]
    x_flat = x.reshape(x.size(0), -1)
    y_flat = y.reshape(y.size(0), -1)
    # Compute cosine similarity along the feature dimension.
    cos_sim = F.cosine_similarity(x_flat, y_flat, dim=1)
    # Cosine distance is defined as 1 - cosine similarity.
    return 1 - cos_sim

def get_metric(metric_string):
    """
    Retrieve the loss metric function based on the provided metric string.

    Args:
        metric_string (str): A string that specifies the desired metric function.
            Valid options are:
                - "CE": Cross Entropy loss metric.
                - "MSE": Mean Squared Error loss metric.
                - "cosine": Cosine distance loss metric.

    Returns:
        function: The corresponding metric function (e.g., CE, MSE, or cosine).

    Raises:
        ValueError: If the metric_string is not one of the valid options.
    """
    if metric_string == "CE":
        return CE
    elif metric_string == "MSE":
        return MSE
    elif metric_string == "cosine":
        return cosine
    else:
        raise ValueError("Invalid metric. Must be 'CE', 'MSE', or 'cosine'.")

def compute_harmonization_loss(images, outputs, labels, clickmaps, has_heatmap, pyramid_levels=5, metric_string=CE, return_saliency=False):
    """
    Computes the harmonization loss as described in Fel's paper.

    This function calculates the gradient-based saliency map, builds Gaussian pyramids for both
    saliency and clickmaps, and then computes a loss over multiple scales.

    Args:
        images (torch.Tensor): Input images of shape [B, C, H, W] with requires_grad=True.
        outputs (torch.Tensor): Model logits of shape [B, num_classes].
        labels (torch.Tensor): Ground truth labels of shape [B].
        clickmaps (list or torch.Tensor): Clickmaps for the batch.
        has_heatmap (list or tensor): Boolean indicator per sample for valid heatmap.
        pyramid_levels (int): Number of levels in the Gaussian pyramid.
        metric (callable): Loss metric to compare pyramid levels (default is CE).
        return_saliency (bool): If True, return a tuple of (loss, saliency_map).

    Returns:
        torch.Tensor or tuple: Harmonization loss, or (loss, saliency_map) if return_saliency is True.
    """
    
    metric = get_metric(metric_string)
    device = images.device
    has_heatmap_bool = torch.tensor(has_heatmap, device=device).float()

    if not has_heatmap_bool.any():
        if return_saliency:
            return torch.tensor(0.0, device=device), None
        return torch.tensor(0.0, device=device)

    if isinstance(clickmaps, list):
        clickmaps = torch.stack(clickmaps, dim=0).to(device)

    batch_size, num_classes = outputs.shape
    one_hot = torch.zeros_like(outputs)
    one_hot.scatter_(1, labels.unsqueeze(1), 1.0)

    gradients = torch.autograd.grad(
        outputs=outputs,
        inputs=images,
        grad_outputs=one_hot,
        create_graph=True 
    )[0]

    valid_saliency = gradients.abs().amax(dim=1, keepdim=True)  # [B, 1, H, W]
    saliency_to_return = valid_saliency.clone() if return_saliency else None

    saliency_pyramid = gaussian_blur(valid_saliency, kernel_size=BRUSH_SIZE, sigma=BRUSH_SIZE_SIGMA)
    clickmap_pyramid = gaussian_blur(clickmaps, kernel_size=BRUSH_SIZE, sigma=BRUSH_SIZE_SIGMA)

    # saliency_pyramid = build_gaussian_pyramid(valid_saliency, levels=pyramid_levels)
    # clickmap_pyramid = build_gaussian_pyramid(clickmaps, levels=pyramid_levels)

    heatmap_count = torch.sum(has_heatmap_bool)

    # if heatmap_count:
    #     ipdb.set_trace()

    # loss_levels = []
    # for level_idx in range(pyramid_levels):
    #     level_differences = metric(saliency_pyramid[level_idx], clickmap_pyramid[level_idx])
    #     level_loss = torch.sum(level_differences * has_heatmap_bool) / heatmap_count
    #     loss_levels.append(level_loss)

    # harmonization_loss = torch.mean(torch.stack(loss_levels))
    harmonization_loss = torch.div(torch.sum(metric(saliency_pyramid, clickmap_pyramid) * has_heatmap_bool), heatmap_count)
    # print(f"HL: {harmonization_loss}, Sum: {torch.sum(metric(saliency_pyramid, clickmap_pyramid)* has_heatmap_bool)}, HC: {heatmap_count}")
    # print("Final Harmonization Loss:", heatmap_count, has_heatmap_bool, harmonization_loss)

    if return_saliency:
        return harmonization_loss, saliency_to_return
    return harmonization_loss

def train_one_epoch(model, dataloader, optimizer, device, epoch, lambda_value=1.0, ce_multiplier=1.0, metric_string=CE):
    """
    Train the model for one epoch.

    For each batch, if any sample has an associated heatmap, this function visualizes the image,
    its saliency map, and its clickmap. The plots are saved to disk in the "plots" directory.
    Other samples are not visualized.

    Additionally, this function logs the average cross entropy loss (cce loss), harmonization loss,
    and total loss to WandB.

    Args:
        model (torch.nn.Module): The model to train.
        dataloader (DataLoader): DataLoader for the training data.
        optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
        device (torch.device): Device to run training on.
        epoch (int): Current epoch number.
        lambda_value (float): Weight for the harmonization loss.

    Returns:
        None
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    pyramid_levels = 5
    running_ce_loss = 0.0
    running_harm_loss = 0.0
    running_total_loss = 0.0
    running_corrects = 0
    total_samples = 0

    # Ensure the plot save directory exists.
    plot_save_dir = "plots"
    os.makedirs(plot_save_dir, exist_ok=True)

    for batch_idx, batch in enumerate(dataloader):
        images, clickmaps, labels, has_heatmap = batch

        #print(f"Batch {batch_idx} Image range: min={images.min().item():.4f}, max={images.max().item():.4f}")
        #print(f"Batch {batch_idx} Clickmap range: min={clickmaps.min().item():.4f}, max={clickmaps.max().item():.4f}, maps:{sum(has_heatmap)}")

        images = images.to(device)
        labels = labels.to(device)
        clickmaps = clickmaps.to(device)
        images.requires_grad_()

        optimizer.zero_grad()

        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            torch.autograd.set_detect_anomaly(True)
            outputs = model(images)

        ce_loss = criterion(outputs, labels)
        # Accumulate ce loss for logging
        running_ce_loss += ce_loss.item()

        preds = outputs.argmax(dim=1)
        running_corrects += (preds == labels).sum().item()
        total_samples += labels.size(0)

        harmonization_loss, saliency_map = compute_harmonization_loss(
            images=images, 
            outputs=outputs, 
            labels=labels, 
            clickmaps=clickmaps, 
            has_heatmap=has_heatmap, 
            pyramid_levels=pyramid_levels,
            return_saliency=True,
            metric_string=metric_string
        )
        running_harm_loss += harmonization_loss.item()

        #if saliency_map is not None:
        #    print(f"Batch {batch_idx} Saliency map range: min={saliency_map.min().item():.4f}, max={saliency_map.max().item():.4f}")

        total_loss = ce_multiplier*ce_loss + lambda_value * harmonization_loss
        running_total_loss += total_loss.item()

        # print("Number of heatmaps:", sum(has_heatmap))
        # ipdb.set_trace()

        total_loss.backward()
        optimizer.step()

        # Visualize and save the plot only for a sample with a valid heatmap.
        # sample_to_plot = None
        # for i, flag in enumerate(has_heatmap):
        #     if flag:
        #         sample_to_plot = i
        #         break

        # if sample_to_plot is not None:
        #     fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        #     # Display the image.
        #     axs[0].imshow(images[sample_to_plot].detach().cpu().permute(1, 2, 0))
        #     axs[0].set_title("Image")
        #     # Display the saliency map.
        #     if saliency_map is not None:
        #         axs[1].imshow(saliency_map[sample_to_plot].detach().cpu().squeeze(0), cmap='viridis')
        #         axs[1].set_title("Saliency Map")
        #     else:
        #         axs[1].text(0.5, 0.5, "No Saliency", horizontalalignment='center')
        #         axs[1].set_title("Saliency Map")
        #     # Display the clickmap.
        #     axs[2].imshow(clickmaps[sample_to_plot].detach().cpu().squeeze(0), cmap='gray')
        #     axs[2].set_title("Clickmap")
        #     # Save the figure.
        #     plot_filename = os.path.join(plot_save_dir, f"epoch{epoch}_batch{batch_idx}_sample{sample_to_plot}.png")
        #     plt.savefig(plot_filename)
        #     plt.close(fig)

        if batch_idx % 10 == 0:
            print(
                f"Epoch {epoch} Batch {batch_idx}: "
                f"CE Loss: {ce_loss.item():.4f}, "
                f"Harm Loss: {harmonization_loss.item():.4f}, "
                f"Total Loss: {total_loss.item():.4f}"
            )

    avg_ce_loss = running_ce_loss / len(dataloader)
    avg_harm_loss = running_harm_loss / len(dataloader)
    avg_total_loss = running_total_loss / len(dataloader)
    accuracy = running_corrects / total_samples
    print(f"Epoch {epoch} Average Training Loss: {avg_total_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    if wandb_logging:
        wandb.log({
            "train_avg_ce_loss": avg_ce_loss, 
            "train_avg_harm_loss": avg_harm_loss, 
            "train_avg_total_loss": avg_total_loss,
            "train_accuracy": accuracy
        })

def validate(model, dataloader, device, pyramid_levels=5):
    """
    Validate the model performance on a validation set.

    Computes cross entropy loss, accuracy, and alignment score (Spearman correlation)
    between saliency maps and clickmaps.

    Additionally, this function logs the average cce loss and the alignment score to WandB.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (DataLoader): DataLoader for validation data.
        device (torch.device): Device to run validation on.
        pyramid_levels (int): Number of pyramid levels for harmonization loss (unused here).

    Returns:
        tuple: (average cce loss, accuracy, average alignment score)
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_ce_loss = 0.0
    running_corrects = 0
    total_samples = 0
    alignment_scores = []

    for batch_idx, batch in enumerate(dataloader):
        images, clickmaps, labels, has_heatmap = batch
        images = images.to(device)
        labels = labels.to(device)
        clickmaps = clickmaps.to(device)
        has_heatmap_bool = torch.tensor(has_heatmap).bool().to(device)

        images.requires_grad_()

        outputs = model(images)
        ce_loss = criterion(outputs, labels)
        running_ce_loss += ce_loss.item()

        preds = outputs.argmax(dim=1)
        running_corrects += (preds == labels).sum().item()
        total_samples += labels.size(0)

        one_hot = torch.zeros_like(outputs)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)

        gradients = torch.autograd.grad(
            outputs=outputs,
            inputs=images,
            grad_outputs=one_hot,
            retain_graph=True,
            allow_unused=True
        )[0]

        saliency = gradients.abs().amax(dim=1, keepdim=True)

        valid_indices = torch.nonzero(has_heatmap_bool).squeeze(1)
        for idx in valid_indices:
            sample_saliency = saliency[idx:idx+1]      
            sample_clickmap = clickmaps[idx:idx+1]       
            alignment = compute_spearman_correlation(sample_saliency, sample_clickmap)
            alignment_scores.append(alignment)

    avg_ce_loss = running_ce_loss / len(dataloader)
    accuracy = running_corrects / total_samples
    avg_alignment = (sum(alignment_scores) / len(alignment_scores)) if alignment_scores else 0.0

    print(f"Validation CCE Loss: {avg_ce_loss:.4f}, Accuracy: {accuracy:.4f}, Alignment Score: {avg_alignment:.4f}")
    if wandb_logging:
        wandb.log({
            "val_avg_ce_loss": avg_ce_loss, 
            "val_accuracy": accuracy, 
            "val_alignment_score": avg_alignment,
        })
    return avg_ce_loss, accuracy, avg_alignment

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

def main():
    """
    Main function to parse arguments, prepare datasets and dataloaders, create the model,
    and run training and validation.
    """
    parser = argparse.ArgumentParser(description='Harmonized ViT Training with ClickMe Dataset')
    parser.add_argument('--epochs', type=int, default=10, help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--train-folder', type=str, default="data/CO3D_ClickMe_Training/", help='training image folder')
    parser.add_argument('--val-folder', type=str, default="data/CO3D_ClickMe_Validation/", help='validation image folder')
    parser.add_argument('--lambda_value', type=float, default=1.0, help='harmonization loss weight')
    parser.add_argument('--ce_multiplier', type=float, default=1.0, help='multiplier for the CE component of the loss')
    parser.add_argument('--metric', type=str, default="CE", help='metric to compute harmonization loss (CE or MSE)')
    

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    label_to_category_map = {}

    train_dataset = ClickMe(args.train_folder, label_to_category_map, is_training=True)
    val_dataset = ClickMe(args.val_folder, label_to_category_map, is_training=False)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = timm.create_model('vit_small_patch16_224.augreg_in21k_ft_in1k', pretrained=True, num_classes=N_CO3D_CLASSES)
    model = nn.DataParallel(model).to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    for epoch in range(1, args.epochs + 1):
        print(f"Starting Epoch {epoch}")
        train_one_epoch(model, train_dataloader, optimizer, device, epoch, args.lambda_value, args.ce_multiplier, args.metric)
        print("Running Validation...")
        validate(model, val_dataloader, device)
        
if __name__ == '__main__':
    main()
