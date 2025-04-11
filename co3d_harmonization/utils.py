import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
import matplotlib.cm as cm
import numpy as np

from .config import KERNEL_SIZE, KERNEL_SIZE_SIGMA

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def gaussian_blur(heatmap, kernel):
    """
    Blurs a heatmap with a Gaussian kernel.

    Parameters
    ----------
    heatmap : torch.Tensor
        The heatmap to blur.
    kernel : torch.Tensor
        The Gaussian kernel.

    Returns
    -------
    blurred_heatmap : torch.Tensor
        The blurred heatmap.
    """
    heatmap = heatmap.to(device)
    kernel = kernel.to(device)

    B, C, H, W = heatmap.shape
    kernel_expanded = kernel.repeat(C, 1, 1, 1)  # shape [C,1,kernel_H,kernel_W]

    blurred = F.conv2d(heatmap, kernel_expanded, padding='same', groups=C)

    return blurred

def get_circle_kernel(size, sigma=None):
    """
    Create a flat circular kernel where the values are the average of the total number of on pixels in the filter.

    Args:
        size (int): The diameter of the circle and the size of the kernel (size x size).
        sigma (float, optional): Not used for flat kernel. Included for compatibility. Default is None.

    Returns:
        torch.Tensor: A 2D circular kernel normalized so that the sum of its elements is 1.
    """
    # Create a grid of (x,y) coordinates
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
    center = (size - 1) / 2
    radius = (size - 1) / 2

    # Create a mask for the circle
    mask = (x - center) ** 2 + (y - center) ** 2 <= radius ** 2

    # Initialize kernel with zeros and set ones inside the circle
    kernel = torch.zeros((size, size), dtype=torch.float32)
    kernel[mask] = 1.0

    # Normalize the kernel so that the sum of all elements is 1
    num_on_pixels = mask.sum()
    if num_on_pixels > 0:
        kernel = kernel / num_on_pixels

    # Add batch and channel dimensions
    kernel = kernel.unsqueeze(0).unsqueeze(0)

    return kernel

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
    circle_kernel = get_circle_kernel(KERNEL_SIZE, KERNEL_SIZE_SIGMA)
    for _ in range(0, levels):
        blurred = gaussian_blur(current, circle_kernel)
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