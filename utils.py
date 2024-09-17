import torch
import numpy as np
from scipy.stats import spearmanr
from torchmetrics.regression import SpearmanCorrCoef

def create_clickmap(point_lists, image_shape, exponential_decay=False, tau=0.5):
    """
    Create a clickmap from click points.

    Args:
        click_points (list of tuples): List of (x, y) coordinates where clicks occurred.
        image_shape (tuple): Shape of the image (height, width).
        blur_kernel (torch.Tensor, optional): Gaussian kernel for blurring. Default is None.
        tau (float, optional): Decay rate for exponential decay. Default is 0.5 but this needs to be tuned.

    Returns:
        np.ndarray: A 2D array representing the clickmap, blurred if kernel provided.
    """
    heatmap = np.zeros(image_shape, dtype=np.uint8)
    for click_points in point_lists:
        if exponential_decay:
            for idx, point in enumerate(click_points):

                if 0 <= point[1] < image_shape[0] and 0 <= point[0] < image_shape[1]:
                    heatmap[point[1], point[0]] += np.exp(-idx / tau)
        else:
            for point in click_points:
                if 0 <= point[1] < image_shape[0] and 0 <= point[0] < image_shape[1]:
                    heatmap[point[1], point[0]] += 1
    return heatmap


def spearman_correlation_np(heatmaps_a: np.ndarray, heatmaps_b: np.ndarray) -> np.ndarray:
    """
    Computes the Spearman correlation between two sets of heatmaps using NumPy.

    Parameters
    ----------
    heatmaps_a : np.ndarray
        First set of heatmaps. Expected shape (N, W, H).
    heatmaps_b : np.ndarray
        Second set of heatmaps. Expected shape (N, W, H).

    Returns
    -------
    np.ndarray
        Array of Spearman correlation scores between the two sets of heatmaps.

    Raises
    ------
    AssertionError
        If the shapes of the input heatmaps are not identical or not (N, W, H).
    """
    assert heatmaps_a.shape == heatmaps_b.shape, "The two sets of heatmaps must have the same shape."
    assert len(heatmaps_a.shape) == 3, "The two sets of heatmaps must have shape (N, W, H)."

    heatmaps_a = _ensure_numpy(heatmaps_a)
    heatmaps_b = _ensure_numpy(heatmaps_b)

    return np.array([spearmanr(ha.flatten(), hb.flatten())[0] for ha, hb in zip(heatmaps_a, heatmaps_b)])

def spearman_correlation(heatmaps_a: torch.Tensor, heatmaps_b: torch.Tensor) -> torch.Tensor:
    """
    Computes the Spearman correlation between two sets of heatmaps using PyTorch.

    Parameters
    ----------
    heatmaps_a : torch.Tensor
        First set of heatmaps. Expected shape (N, W, H).
    heatmaps_b : torch.Tensor
        Second set of heatmaps. Expected shape (N, W, H).

    Returns
    -------
    torch.Tensor
        Tensor of Spearman correlation scores between the two sets of heatmaps.

    Raises
    ------
    AssertionError
        If the shapes of the input heatmaps are not identical or not (N, W, H).
    """
    assert heatmaps_a.shape == heatmaps_b.shape, "The two sets of heatmaps must have the same shape."
    assert len(heatmaps_a.shape) == 3, "The two sets of heatmaps must have shape (N, W, H)."

    batch_size = heatmaps_a.shape[0]
    spearman = SpearmanCorrCoef(num_outputs=1)
    heatmaps_a = heatmaps_a.reshape(batch_size, -1)
    heatmaps_b = heatmaps_b.reshape(batch_size, -1)

    return torch.stack([spearman(heatmaps_a[i], heatmaps_b[i]) for i in range(batch_size)])

def compute_human_alignment(predicted_heatmaps: torch.Tensor, clickme_heatmaps: torch.Tensor) -> float:
    """
    Computes the human alignment score between predicted heatmaps and ClickMe heatmaps.

    Parameters
    ----------
    predicted_heatmaps : torch.Tensor
        Predicted heatmaps. Expected shape (N, C, W, H) or (N, W, H).
    clickme_heatmaps : torch.Tensor
        ClickMe heatmaps. Expected shape (N, C, W, H) or (N, W, H).

    Returns
    -------
    float
        Human alignment score.
    """
    HUMAN_SPEARMAN_CEILING = 0.65753

    predicted_heatmaps = _ensure_3d(predicted_heatmaps)
    clickme_heatmaps = _ensure_3d(clickme_heatmaps)

    scores = spearman_correlation(predicted_heatmaps, clickme_heatmaps)
    human_alignment = scores.mean().item() / HUMAN_SPEARMAN_CEILING

    return human_alignment

def _ensure_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Ensures the input tensor is a NumPy array.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor to convert.

    Returns
    -------
    np.ndarray
        NumPy array version of the input tensor.
    """
    if isinstance(tensor, np.ndarray):
        return tensor
    return tensor.cpu().numpy()

def _ensure_3d(heatmaps: torch.Tensor) -> torch.Tensor:
    """
    Ensures the input heatmaps are 3D by removing the channel dimension if present.

    Parameters
    ----------
    heatmaps : torch.Tensor
        Input heatmaps. Expected shape (N, C, W, H) or (N, W, H).

    Returns
    -------
    torch.Tensor
        3D heatmaps with shape (N, W, H).
    """
    if len(heatmaps.shape) == 4:
        return heatmaps[:, 0, :, :]
    return heatmaps

def gaussian_kernel(size=10, sigma=10):
    """
    Generates a 2D Gaussian kernel.

    Parameters
    ----------
    size : int, optional
        Kernel size, by default 10
    sigma : int, optional
        Kernel sigma, by default 10

    Returns
    -------
    kernel : torch.Tensor
        A Gaussian kernel.
    """
    x_range = torch.arange(-(size-1)//2, (size-1)//2 + 1, 1)
    y_range = torch.arange((size-1)//2, -(size-1)//2 - 1, -1)

    xs, ys = torch.meshgrid(x_range, y_range, indexing='ij')
    kernel = torch.exp(-(xs**2 + ys**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)
    
    kernel = kernel / kernel.sum()
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    
    return kernel

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
    # Ensure heatmap and kernel have the correct dimensions
    heatmap = heatmap.unsqueeze(0) if heatmap.dim() == 3 else heatmap
    blurred_heatmap = torch.nn.functional.conv2d(heatmap, kernel, padding='same')

    return blurred_heatmap[0]