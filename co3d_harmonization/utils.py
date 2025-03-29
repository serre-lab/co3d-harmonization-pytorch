import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
import matplotlib.cm as cm
import numpy as np

from .config import BRUSH_SIZE, BRUSH_SIZE_SIGMA

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

def min_max_normalize_maps(tensor):
    """
    Normalizes each image using min-max, handling either 3D (C,H,W) or 4D (B,C,H,W).
    """
    epsilon = 1e-8

    if tensor.ndim == 3:
        # shape = (C,H,W) => min/max over (1,2)
        min_vals = tensor.amin(dim=(1,2), keepdim=True)
        max_vals = tensor.amax(dim=(1,2), keepdim=True)
        return (tensor - min_vals) / (max_vals - min_vals + epsilon)

    elif tensor.ndim == 4:
        # shape = (B,C,H,W) => min/max over (2,3)
        min_vals = tensor.amin(dim=(2,3), keepdim=True)
        max_vals = tensor.amax(dim=(2,3), keepdim=True)
        return (tensor - min_vals) / (max_vals - min_vals + epsilon)
    
    else:
        raise ValueError(f"Unsupported tensor shape {tensor.shape}, expected 3D or 4D.")

def stack_channels(t):
        """Convert a [H,W] or [H,W,C=1] tensor into [H,W,3]."""
        if t.ndim == 2: 
            # [H, W] -> [H, W, 1]
            t = t.unsqueeze(-1)
        if t.shape[-1] == 1:
            # reapet the single channel 3 times
            t = t.repeat(1, 1, 3)
        return t

def apply_colormap(tensor, cmap="viridis"):
    """
    Given a 2D PyTorch tensor in [0..1] (H, W), 
    return an [H, W, 3] NumPy array colored by the chosen colormap.
    """
    # Convert to CPU and NumPy
    array = tensor.detach().cpu().numpy()

    # Apply the colormap (returns RGBA
    colored_image = cm.get_cmap(cmap)(array)  # shape [H, W, 4]

    # Discard the alpha channel -> [H, W, 3]
    colored_image = colored_image[..., :3]

    # Tescale to [0..255] if you want uint8
    colored_image = (colored_image * 255).astype(np.uint8)

    return colored_image

def denormalize_image(tensor):
    """
    Invert the normalization that was done by torchvision.transforms.Normalize.
    
    Args:
        tensor (torch.Tensor): The image of shape [H, W, C] or [C, H, W], typically in the range [-1, 1].
        mean, std (torch.Tensor): Tensors of shape [3] for RGB.
    
    Returns:
        torch.Tensor in range [0,1], shape the same as 'tensor'.
    """
    mean = torch.tensor([0.5, 0.5, 0.5])
    std  = torch.tensor([0.5, 0.5, 0.5])

    if tensor.ndim == 3 and tensor.shape[-1] == 3:
        # x_denorm = x_norm * std + mean
        tensor = tensor * std + mean   # shape still [H, W, 3]

    elif tensor.ndim == 3 and tensor.shape[0] == 3:
        # x_denorm = x_norm * std[:,None,None] + mean[:,None,None]
        tensor = tensor * std[:, None, None] + mean[:, None, None]
        
    else:
        raise ValueError(f"Tensor shape not recognized for denormalization: {tensor.shape}")

    # clamp to [0..1]
    tensor = torch.clamp(tensor, 0.0, 1.0)
    return tensor