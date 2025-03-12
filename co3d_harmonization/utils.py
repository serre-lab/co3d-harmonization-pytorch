import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

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