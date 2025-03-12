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

def _get_metric(metric_string):
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
    
    metric = _get_metric(metric_string)
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
    # calculate z-score per-image
    # mask = z_valid_saliency >= 0
    # Compute loss on non-masked values

    # ipdb.set_trace()
    
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

    # UNCOMMENTED FROM HERE
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

    # # Z-score normalization 
    # # Compute per-image z-score
    # B, _, H, W = valid_saliency.shape
    # valid_saliency_flat = valid_saliency.view(B, -1)
    # mean = valid_saliency_flat.mean(dim=1, keepdim=True)
    # std = valid_saliency_flat.std(dim=1, keepdim=True) + 1e-7
    #  # Normalize
    # z_valid_saliency = (valid_saliency_flat - mean) / std
    # # Reshape back to [B, 1, H, W]
    # z_valid_saliency = z_valid_saliency.view(B, 1, H, W) 
    # # print("Nomalized shape:", z_valid_saliency.shape)

    # # Apply ReLU so that valuesbelow mean = 0
    # # z_valid_saliency = torch.relu(z_valid_saliency)
    # # # valid pixcel mask
    # # mask = (z_valid_saliency > 0).float()
    # # masked_saliency = z_valid_saliency * mask
    # # masked_clickmaps = clickmaps * mask # Should we mask the clickmaps here

    # # Optionally return the z-normalized saliency instead of the raw one.
    # saliency_to_return = z_valid_saliency.clone() if return_saliency else None

    # saliency_pyramid = gaussian_blur(z_valid_saliency, kernel_size=BRUSH_SIZE, sigma=BRUSH_SIZE_SIGMA)
    # clickmap_pyramid = gaussian_blur(clickmaps, kernel_size=BRUSH_SIZE, sigma=BRUSH_SIZE_SIGMA)

    # # saliency_pyramid = build_gaussian_pyramid(valid_saliency, levels=pyramid_levels)
    # # clickmap_pyramid = build_gaussian_pyramid(clickmaps, levels=pyramid_levels)

    # heatmap_count = torch.sum(has_heatmap_bool)

    # # if heatmap_count:
    # #     ipdb.set_trace()

    # # loss_levels = []
    # # for level_idx in range(pyramid_levels):
    # #     level_differences = metric(saliency_pyramid[level_idx], clickmap_pyramid[level_idx])
    # #     level_loss = torch.sum(level_differences * has_heatmap_bool) / heatmap_count
    # #     loss_levels.append(level_loss)

    # # TODO: Check k_masks here

    # # normalized_saliency = torch.sigmoid(saliency_pyramid)

    # # harmonization_loss = (F.binary_cross_entropy(normalized_saliency, clickmap_pyramid, reduction='none') * k_masks).reshape(len(normalized_saliency), -1)
    # # harmonization_loss = harmonization_loss.sum(1) / k_masks.reshape(len(k_masks), -1).sum(1)
    # # harmonization_loss = harmonization_loss.mean()plots
    # # harmonization_loss = torch.mean(torch.stack(loss_levels))
    # harmonization_loss = torch.div(torch.sum(metric(saliency_pyramid, clickmap_pyramid) * has_heatmap_bool), heatmap_count)
    # # print(f"HL: {harmonization_loss}, Sum: {torch.sum(metric(saliency_pyramid, clickmap_pyramid)* has_heatmap_bool)}, HC: {heatmap_count}")
    # # print("Final Harmonization Loss:", heatmap_count, has_heatmap_bool, harmonization_loss)

    # if return_saliency:
    #     return harmonization_loss, saliency_to_return
    # return harmonization_loss
