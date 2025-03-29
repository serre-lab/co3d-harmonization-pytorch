# losses.py
import torch
import torch.nn.functional as F

from .utils import *
from .config import BRUSH_SIZE, BRUSH_SIZE_SIGMA

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

def BCE(x, y):
    """
    Compute Binary Cross Entropy (BCE) loss between two tensors on a per-pixel basis.

    Args:
        x (torch.Tensor): Predicted tensor (expected to be probabilities).
        y (torch.Tensor): Target tensor (expected to be in the same range as x).

    Returns:
        torch.Tensor: Per-sample BCE loss computed over flattened dimensions.
    """
    x_flat = x.reshape(x.size(0), -1)
    y_flat = y.reshape(y.size(0), -1)
    return F.binary_cross_entropy(x_flat, y_flat, reduction='none').mean(dim=1)

def _get_metric(metric_string):
    """
    Retrieve the loss metric function based on the provided metric string.

    Args:
        metric_string (str): A string that specifies the desired metric function.
            Valid options are:
                - "CE": Cross Entropy loss metric.
                - "MSE": Mean Squared Error loss metric.
                - "cosine": Cosine distance loss metric.
                - "BCE": Binary Cross Entropy loss metric.

    Returns:
        function: The corresponding metric function (e.g., CE, MSE, cosine, or BCE).

    Raises:
        ValueError: If the metric_string is not one of the valid options.
    """
    if metric_string == "CE":
        return CE
    elif metric_string == "MSE":
        return MSE
    elif metric_string == "cosine":
        return cosine
    elif metric_string == "BCE":
        return BCE
    else:
        raise ValueError("Invalid metric. Must be 'CE', 'MSE', 'cosine', or 'BCE'.")

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
    metric = _get_metric(metric_string)
    device = images.device
    has_heatmap_bool = torch.tensor(has_heatmap, device=device).bool()

    # 1) If no sample has a heatmap, return 0
    if not has_heatmap_bool.any():
        return (torch.tensor(0.0, device=device), None) if return_saliency \
               else torch.tensor(0.0, device=device)

    # 2) Convert clickmaps to a tensor
    if isinstance(clickmaps, list):
        clickmaps = torch.stack(clickmaps, dim=0).to(device)

    # 3) Build the one-hot for the true class
    batch_size, num_classes = outputs.shape
    one_hot = torch.zeros_like(outputs)
    one_hot.scatter_(1, labels.unsqueeze(1), 1.0)

    # 4) Gradients wrt images -> saliency
    gradients = torch.autograd.grad(
        outputs=outputs,
        inputs=images,
        grad_outputs=one_hot,
        create_graph=True
    )[0]

    # 5) Absolute saliency value then max across channel dimension
    saliency = gradients.abs().amax(dim=1, keepdim=True)  # shape [B, 1, H, W]

    # Min-max normalize both maps
    saliency_norm = min_max_normalize_maps(saliency)
    clickmaps_norm = min_max_normalize_maps(clickmaps)

    # --- Z-score normalization for saliency map per image ---
    # B, C, H, W = valid_saliency.shape
    # valid_saliency_flat = valid_saliency.view(B, -1)
    # mean = valid_saliency_flat.mean(dim=1, keepdim=True)
    # std = valid_saliency_flat.std(dim=1, keepdim=True) + 1e-7
    # z_valid_saliency = (valid_saliency_flat - mean) / std
    # z_valid_saliency = z_valid_saliency.view(B, C, H, W)
    # valid_saliency = z_valid_saliency

    saliency_to_return = saliency_norm.clone() if return_saliency else None

    # Adding Gaussian Blur
    saliency_pyramid = gaussian_blur(saliency_norm, kernel_size=BRUSH_SIZE, sigma=BRUSH_SIZE_SIGMA)
    clickmap_pyramid = gaussian_blur(clickmaps_norm, kernel_size=BRUSH_SIZE, sigma=BRUSH_SIZE_SIGMA)
    # saliency_pyramid = build_gaussian_pyramid(saliency_norm, levels=pyramid_levels)
    # clickmap_pyramid = build_gaussian_pyramid(clickmaps_norm, levels=pyramid_levels)

    heatmap_count = torch.sum(has_heatmap_bool)
    
    harmonization_loss = torch.div(torch.sum(metric(saliency_pyramid, clickmap_pyramid) * has_heatmap_bool), heatmap_count)
    # TODO: Add the gaussian pyramid back in
    # Using Gaussian Pyramid
    # loss_levels = []
    # for level_idx in range(pyramid_levels):
    #     level_differences = metric(saliency_pyramid[level_idx], clickmap_pyramid[level_idx])
    #     level_loss = torch.sum(level_differences * has_heatmap_bool) / heatmap_count
    #     loss_levels.append(level_loss)
    # harmonization_loss = torch.mean(torch.stack(loss_levels))

    if return_saliency:
        return harmonization_loss, saliency_to_return
    return harmonization_loss

    # Do the k-mask for each training image at an image-level
    # K value for each map
    # k_clickmaps = (clickmaps > 0).float().reshape(len(clickmaps), -1).sum(axs=-1)

    # # for each image get the K
    # threshholds = []
    # k_masks = []
    # for cm, sal in zip(k_clickmaps, valid_saliency):
    #     current_thresh = torch.sort(sal, descending=True)[cm]
    #     k_masks.append((sal > current_thresh))

    # k_masks = torch.stack(k_masks, dim=0).float()

    # Apply ReLU so that valuesbelow mean = 0
    # z_valid_saliency = torch.relu(z_valid_saliency)
    # # valid pixcel mask
    # mask = (z_valid_saliency > 0).float()
    # masked_saliency = z_valid_saliency * mask
    # masked_clickmaps = clickmaps * mask # Should we mask the clickmaps here
