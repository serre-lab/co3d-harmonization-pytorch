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
import wandb    
from torchvision.utils import save_image


N_CO3D_CLASSES = 51
BRUSH_SIZE = 11
BRUSH_SIZE_SIGMA = np.sqrt(BRUSH_SIZE)
HUMAN_SPEARMAN_CEILING = 0.4422303328731989


def save_debug_images(saliency_pyramid, clickmap_pyramid, levels=[0, 1], num_samples=4):
    """
    Saves sample images from the saliency and clickmap pyramids for debugging.
    
    Args:
        saliency_pyramid: list of tensors, each of shape [B, 1, H, W]
        clickmap_pyramid: list of tensors, each of shape [B, 1, H, W]
        levels: list of pyramid levels to save (default: levels 0 and 1)
        num_samples: number of samples from the batch to save from each level
    """
    for level in levels:
        # Save a grid of the first num_samples images
        save_image(clickmap_pyramid[level][:num_samples], f"debug_clickmap_level{level}.png", normalize=True)
        save_image(saliency_pyramid[level][:num_samples], f"debug_model_map_level{level}.png", normalize=True)


def get_gaussian_kernel(kernel_size=BRUSH_SIZE, sigma=BRUSH_SIZE_SIGMA, channels=1):
    ax = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size // 2)
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    return kernel.repeat(channels, 1, 1, 1)

class utils:
    @staticmethod
    def gaussian_kernel(size, sigma):
        return get_gaussian_kernel(kernel_size=size, sigma=sigma, channels=1)


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

# -------------------- ClickMe Dataset --------------------
class ClickMe(Dataset):
    """
    A custom Dataset class for the ClickMe dataset.
    Loads image, heatmap, and label data from file paths.
    """
    def __init__(self, image_folder, label_to_category_map, is_training=True):
        super().__init__()
        self.image_folder = image_folder
        self.label_to_category_map = label_to_category_map
        self.is_training = is_training
        self.data = []
        self.data_dictionary = {}

        if is_training:
            image_path = "/cifs/data/tserre_lrs/projects/projects/prj_video_imagenet/CO3D_ClickMe_Training/"
            co3d_clickme = pd.read_csv("data/CO3D_ClickMe_Training.csv")
            output_dir = "assets"
            image_output_dir = "clickme_test_images"
            image_shape = [256, 256]
            exponential_decay = False

            category_index = 0
            os.makedirs(image_output_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)

            # Process images WITHOUT ClickMaps.
            text_file = "/cifs/data/tserre_lrs/projects/projects/prj_video_imagenet/CausalVisionModeling/data_lists/filtered_binocular_renders_test.txt"
            root_dir = "/cifs/data/tserre_lrs/projects/projects/prj_video_imagenet/PeRFception/data/co3d_v2/"
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
                    for i in range(0, 50, 9):
                        full_path = os.path.join(root_dir, path, files[i].strip())
                        image_name = "_".join(full_path.split("/")[-4:])
                        label = full_path.split("/")[-4]
                        self.data.append({
                            'image': full_path,
                            'heatmap': torch.from_numpy(np.zeros((256, 256))).float(),
                            'category_label': self.label_to_category_map[label],
                            'has_heatmap': False,
                        })
            print("Done processing training images WITHOUT ClickMaps. There are", len(self.data), "images.")

            # Process images WITH ClickMaps.
            processed_maps, num_maps = process_clickmaps(co3d_clickme, is_training=is_training)
            gaussian_kernel = utils.gaussian_kernel(size=BRUSH_SIZE, sigma=BRUSH_SIZE_SIGMA)
            heatmap_count = 0
            for idx, (image, maps) in enumerate(processed_maps.items()):
                full_path = os.path.join(image_path, image)
                image_name, image_obj, heatmap = make_heatmap(
                    os.path.join(image_path, image), maps, gaussian_kernel,
                    image_shape=image_shape, exponential_decay=exponential_decay)
                label = image_name.split("/")[-2]
                if image_name is None:
                    continue
                image_name = "_".join(image_name.split("/")[-2:])
                self.data.append({
                    "image": full_path,
                    "heatmap": heatmap,
                    "category_label": self.label_to_category_map[label],
                    "has_heatmap": True
                })
                heatmap_count += 1
            print("Done processing training images WITH ClickMaps. There are", heatmap_count, "heatmaps.")
        else:
            image_path = "/cifs/data/tserre_lrs/projects/projects/prj_video_imagenet/CO3D_ClickMe2/"
            co3d_clickme = pd.read_csv("data/CO3D_ClickMe_Validation.csv")
            output_dir = "assets"
            image_output_dir = "clickme_test_images"
            image_shape = [256, 256]
            exponential_decay = False

            category_index = 0
            os.makedirs(image_output_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)

            processed_maps, num_maps = process_clickmaps(co3d_clickme, is_training=is_training)
            gaussian_kernel = utils.gaussian_kernel(size=BRUSH_SIZE, sigma=BRUSH_SIZE_SIGMA)
            for idx, (image, maps) in enumerate(processed_maps.items()):
                full_path = os.path.join(image_path, image)
                image_name, image_obj, heatmap = make_heatmap(
                    os.path.join(image_path, image), maps, gaussian_kernel,
                    image_shape=image_shape, exponential_decay=exponential_decay)
                label = image_name.split("/")[2]
                if label not in self.label_to_category_map.keys():
                    self.label_to_category_map[label] = category_index
                    category_index += 1
                if image_name is None:
                    continue
                image_name = "_".join(image_name.split("/")[-2:])
                self.data.append({
                    "image": full_path,
                    "heatmap": heatmap,
                    "category_label": self.label_to_category_map[label],
                    "has_heatmap": True
                })

    def __getitem__(self, index):
        center_crop = [224, 224]
        img_path = self.data[index]['image']
        hmp = self.data[index]['heatmap']
        label = self.data[index]['category_label']
        has_heatmap = self.data[index]['has_heatmap']
        image_name = img_path 
        img = Image.open(img_path).convert("RGB")
        img = tvF.center_crop(img, center_crop)
        img = self._preprocess_image(img)
        hmp = tvF.center_crop(hmp, center_crop)
        hmp = self._preprocess_heatmap(hmp)
        label = self._preprocess_label(label)
        if self.is_training:
            img, hmp = self._apply_augmentation(img, hmp)
        img = data_transforms['norm'](img)
        return img, hmp, label, has_heatmap

    def __len__(self):
        return len(self.data)

    def _preprocess_image(self, img):
        img_array = np.array(img)
        img_tensor = torch.from_numpy(img_array)
        img_tensor = img_tensor.permute(2, 0, 1)
        return img_tensor.to(torch.float32) / 255.0

    def _preprocess_heatmap(self, hmp):
        if not torch.is_tensor(hmp):
            hmp = torch.tensor(np.array(hmp))
        hmp = hmp.to(torch.float32) / 255.0
        if hmp.ndim == 2:  # Ensure heatmap has a channel dimension.
            hmp = hmp.unsqueeze(0)
        return hmp

    def _preprocess_label(self, label):
        label = torch.tensor(label)
        return torch.squeeze(label.to(torch.int64))

    def _apply_augmentation(self, img, hmp):
        # img is [3, H, W] and hmp is [1, H, W]
        stacked_img = torch.cat((img, hmp), dim=0)  # shape: [4, H, W]
        stacked_img = data_transforms['aug'](stacked_img)
        return stacked_img[:3, :, :], stacked_img[3:, :, :]  # hmp remains [1, H, W]

def gaussian_blur(x, kernel_size=5, sigma=1.0):
    channels = x.shape[1]
    kernel = get_gaussian_kernel(kernel_size, sigma, channels).to(x.device)
    padding = kernel_size // 2
    return F.conv2d(x, kernel, padding=padding, groups=channels)

def build_gaussian_pyramid(x, levels=5):
    pyramid = [x]
    current = x
    for _ in range(1, levels):
        blurred = gaussian_blur(current, kernel_size=5, sigma=1.0)
        downsampled = F.avg_pool2d(blurred, kernel_size=2, stride=2)
        pyramid.append(downsampled)
        current = downsampled
    return pyramid

def compute_spearman_correlation(saliency, clickmap):
    saliency_np = saliency.view(-1).detach().cpu().numpy()
    clickmap_np = clickmap.view(-1).detach().cpu().numpy()
    correlation, _ = spearmanr(saliency_np, clickmap_np)
    return correlation

def MSE(x, y):
    x = x.reshape(len(x), -1)
    y = y.reshape(len(y), -1)
    return torch.square(x-y).mean(-1)

def CE(q, k):
    q = q.reshape(len(q), -1)
    k = F.softmax(k.reshape(len(k), -1), dim=-1)
    return F.cross_entropy(q, k, reduction='none')

def compute_harmonization_loss(images, outputs, labels, clickmaps, has_heatmap, pyramid_levels=5, metric=MSE):
    """
    Computes the harmonization loss from Fel's paper

    Args:
        images:         [B, C, H, W]  input images (with requires_grad=True)
        outputs:        [B, num_classes]  model logits for entire batch
        labels:         [B] integer labels for classification
        clickmaps:      list or tensor of length B containing clickmaps
        has_heatmap:    [B] bool indicating whether an image has a valid heatmap
        pyramid_levels: int, number of Gaussian pyramid levels

    Returns:
        A single scalar tensor (the harmonization loss).
    """
    device = images.device
    has_heatmap_bool = torch.tensor(has_heatmap).float()

    # If no images in this batch have a valid heatmap, skip
    if not has_heatmap_bool.any():
        return torch.tensor(0.0, device=device)
    
    # 1. Create a one-hot version of the classification targets
    batch_size, num_classes = outputs.shape
    one_hot = torch.zeros_like(outputs)  # [B, num_classes]
    one_hot.scatter_(1, labels.unsqueeze(1), 1.0)

    # 2. Zero out one-hot entries where has_heatmap is False
    # one_hot[~has_heatmap_bool] = 1.0
    
    # 3. Compute saliency by taking gradient of outputs wrt the input images
    gradients = torch.autograd.grad(
        outputs=outputs,        # [B, numm_classes]
        inputs=images,          #  [B, C, H, W]
        grad_outputs=one_hot,   # [B, num_classes]
        create_graph=True 
    )[0]
    
    # 4. Convert gradient to saliency
    valid_saliency = gradients.abs().amax(dim=1, keepdim=True)  # [B, 1, H, W]
    saliency_pyramid = build_gaussian_pyramid(valid_saliency, levels=pyramid_levels)
    clickmap_pyramid = build_gaussian_pyramid(clickmaps, levels=pyramid_levels)

    has_heatmap_tensor = torch.tensor(has_heatmap, dtype=torch.int, device=device)
    heatmap_count = torch.sum(has_heatmap_tensor)
    # 7. Compute MSE across pyramid levels
    loss_levels = []
    for level_idx in range(pyramid_levels):
        level_differences = metric(saliency_pyramid[level_idx], clickmap_pyramid[level_idx])
        level_loss = torch.sum(level_differences * has_heatmap_tensor)/heatmap_count

        loss_levels.append(level_loss)
        # Debug: Save sample images from a couple of pyramid levels
        # save_debug_images(saliency_pyramid, clickmap_pyramid, levels=[0, 1], num_samples=4)
    
    # Average across levels
    harmonization_loss = 1e7 * torch.mean(torch.tensor(loss_levels))

    return harmonization_loss

    # loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    lambda1 = 1.0  # Harmonization loss weight
    pyramid_levels = 5
    running_loss = 0.0
    for batch_idx, batch in enumerate(dataloader):
        images, clickmaps, labels, has_heatmap = batch
        images = images.to(device)
        labels = labels.to(device)
        clickmaps = torch.stack(clickmaps)
        images.requires_grad_()

        optimizer.zero_grad()

        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            torch.autograd.set_detect_anomaly(True)
            outputs = model(images)
        ce_loss = criterion(outputs, labels)
        # Harmonization loss 
        harmonization_loss = compute_harmonization_loss(images=images, outputs=outputs, labels=labels, clickmaps=clickmaps, has_heatmap=has_heatmap, pyramid_levels=pyramid_levels)

        # Combine losses
        total_loss = ce_loss + lambda1 * harmonization_loss
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()

        # Debug print
        if batch_idx % 10 == 0:
            print(
                f"Epoch {epoch} Batch {batch_idx}: "
                f"CE Loss: {ce_loss.item():.4f}, "
                f"Harm Loss: {harmonization_loss.item():.4f}, "
                f"Total Loss: {total_loss.item():.4f}"
            )

    avg_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch} Average Training Loss: {avg_loss:.4f}")
    if wandb_logging:
        wandb.log({"train_loss": train_loss/len(train_loader),  "cce_loss": ce_loss.item(), "harmonization_loss": harmonization_loss.item()})

def validate(model, dataloader, device, pyramid_levels=5):
    model.eval()
    alignment_scores = []
    total_loss = 0.0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in dataloader:
            images, clickmaps, labels, has_heatmap = batch
            has_heatmap = list(has_heatmap)
            images = images.to(device)
            labels = labels.to(device)
            clickmaps = torch.stack(clickmaps)
            images.requires_grad_()

            outputs = model(images)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            for i in range(images.shape[0]):
                if has_heatmap[i]:
                    output = outputs[i:i+1]
                    label = labels[i:i+1]
                    one_hot = torch.zeros_like(output)
                    one_hot[0, label] = 1.0
                    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                        torch.autograd.set_detect_anomaly(True)
                        grad = torch.autograd.grad(outputs=output, inputs=images[i:i+1], grad_outputs=one_hot, retain_graph=True)[0]
                    saliency = grad.abs().mean(dim=1, keepdim=True)
                    alignment = compute_spearman_correlation(saliency, clickmaps[i].to(device))
                    alignment_scores.append(alignment)

    avg_loss = total_loss / len(dataloader)
    avg_alignment = sum(alignment_scores) / len(alignment_scores) if alignment_scores else 0.0

    print(f"Validation Loss: {avg_loss:.4f}, Alignment Score: {avg_alignment:.4f}")
    return avg_loss, avg_alignment

def collate_fn(batch):
    images, heatmaps, labels, image_names = zip(*batch)
    images = torch.stack(images, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    heatmaps = list(heatmaps)
    return images, heatmaps, labels, image_names

def main():
    parser = argparse.ArgumentParser(description='Harmonized ViT Training with ClickMe Dataset')
    parser.add_argument('--epochs', type=int, default=2, help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--train-folder', type=str, default="data/CO3D_ClickMe_Training/", help='training image folder')
    parser.add_argument('--val-folder', type=str, default="data/CO3D_ClickMe_Validation/", help='validation image folder')
    args = parser.parse_args()

    wandb_logging = False

    if wandb_logging:
        wandb.init(entity="jayrgopal", project="harmonization",
        config={
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "train_folder": args.train_folder,
        "val_folder": args.val_folder
    })

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    label_to_category_map = {}

    train_dataset = ClickMe(args.train_folder, label_to_category_map, is_training=True)
    val_dataset = ClickMe(args.val_folder, label_to_category_map, is_training=False)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=N_CO3D_CLASSES)
    model = nn.DataParallel(model).to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    for epoch in range(1, args.epochs + 1):
        print(f"Starting Epoch {epoch}")
        train_one_epoch(model, train_dataloader, optimizer, device, epoch)
        print("Running Validation...")
        validate(model, val_dataloader, device)
        
if __name__ == '__main__':
    main()