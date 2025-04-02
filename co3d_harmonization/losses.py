# losses.py
import torch
import torch.nn.functional as F

from .utils import get_circle_kernel, gaussian_blur, min_max_normalize_maps, build_gaussian_pyramid
from .config import KERNEL_SIZE, KERNEL_SIZE_SIGMA

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

def BCE_masked(x, y, mask):
    """
    Compute BCE loss per sample over only pixels where mask is 1.
    
    Args:
        x (torch.Tensor): Predicted probabilities in [0,1], shape [B, H, W].
        y (torch.Tensor): Target tensor in [0,1], shape [B, H, W].
        mask (torch.Tensor): Binary mask (0/1) of shape [B, H, W].
    
    Returns:
        torch.Tensor: Per-sample BCE loss, averaged over valid pixels.
    """
    # Flatten
    x_flat = x.reshape(x.size(0), -1)
    y_flat = y.reshape(y.size(0), -1)
    
    # Clamp x to avoid log(0) in BCE
    x_flat = x_flat.clamp(min=1e-8, max=1 - 1e-8)
    y_flat = y_flat.clamp(min=1e-8, max=1 - 1e-8)
    
    mask_flat = mask.reshape(mask.size(0), -1)
    
    # Compute per-pixel BCE loss
    loss = F.binary_cross_entropy(x_flat, y_flat, reduction='none')
    # Zero out loss wherever the mask==0
    loss = loss * mask_flat
    # QAverage over valid pixels
    valid_counts = mask_flat.sum(dim=1) + 1e-8
    loss_per_sample = loss.sum(dim=1) / valid_counts
    return loss_per_sample

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
        return BCE_masked
    else:
        raise ValueError("Invalid metric. Must be 'CE', 'MSE', 'cosine', or 'BCE'.")

def compute_harmonization_loss(model, images, outputs, labels, clickmaps, has_heatmap, pyramid_levels=5, metric_string=CE, return_saliency=False, get_top_k=False):
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
    circle_kernel = get_circle_kernel(KERNEL_SIZE, KERNEL_SIZE_SIGMA)

    metric = _get_metric(metric_string)
    device = images.device
    has_heatmap_bool = torch.tensor(has_heatmap, device=device).bool()

    # If no sample has a heatmap, return 0
    if not has_heatmap_bool.any():
        return (torch.tensor(0.0, device=device), None) if return_saliency \
               else torch.tensor(0.0, device=device)

    # Convert clickmaps to a tensor
    if isinstance(clickmaps, list):
        clickmaps = torch.stack(clickmaps, dim=0).to(device)

    # Build the one-hot for the true class
    batch_size, num_classes = outputs.shape
    one_hot = torch.zeros_like(outputs)
    one_hot.scatter_(1, labels.unsqueeze(1), 1.0)

    # Gradients wrt images -> saliency
    gradients = torch.autograd.grad(
        outputs=outputs,
        inputs=images,
        grad_outputs=one_hot,
        create_graph=True
    )[0]

    # Absolute saliency value then max across channel dimension
    saliency = gradients.abs().amax(dim=1, keepdim=True)  # shape [B, 1, H, W]

    # import ipdb; ipdb.set_trace()

    # Apply Gaussian blur twice
    saliency = gaussian_blur(saliency, circle_kernel)
    saliency = gaussian_blur(saliency, circle_kernel)
    saliency_final = min_max_normalize_maps(saliency)#.squeeze(1)

    # Project the Saliency
    projected_saliency = model.module.saliency_projection(saliency_final)

    if get_top_k:
        # Compute top-k saliency per sample
        top_k_saliency = projected_saliency.clone()  # [B, 1, H, W]
        B, _, H, W = top_k_saliency.shape
        top_k_saliency = top_k_saliency.view(B, -1)  # [B, H*W]
        clickmaps_flat = clickmaps.view(B, -1)       # [B, H*W]
        for i in range(B):
            if has_heatmap_bool[i]:
                # Count non-zero pixels in the i-th clickmap as k_i
                k_i = (clickmaps_flat[i] > 0).sum().item()
                if k_i > 0:
                    sal_i = top_k_saliency[i]
                    top_vals, _ = torch.topk(sal_i, k_i)
                    threshold_i = top_vals[-1]
                    sal_i[sal_i < threshold_i] = 0.0
        projected_saliency = top_k_saliency.view(B, 1, H, W)

    # Normalize heatmaps
    clickmaps = min_max_normalize_maps(clickmaps)
    
    saliency_logits = projected_saliency.squeeze(1)

    # # Mask to only consider pixels where the raw blurred saliency > 0
    mask = (clickmaps.squeeze(1) > 0).float()
    
    # Prepare the target clickmap (also shape [B, H, W])
    if clickmaps.ndim == 4 and clickmaps.shape[1] == 1:
        target = clickmaps.squeeze(1)
    else:
        target = clickmaps
    
    # Compute BCE loss only on the valid pixels
    loss_per_sample = BCE_masked(saliency_logits, target, mask)
    harmonization_loss = loss_per_sample.mean()
    saliency_to_return = saliency_final.clone() if return_saliency else None
    
    # heatmap_count = torch.sum(has_heatmap_bool)

    if return_saliency:
        return harmonization_loss, saliency_to_return
    return harmonization_loss



def compute_FEL_harmonization_loss(model, images, outputs, labels, clickmaps, has_heatmap, pyramid_levels=5, metric_string=CE, return_saliency=False, get_top_k=False):
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
    circle_kernel = get_circle_kernel(KERNEL_SIZE, KERNEL_SIZE_SIGMA)

    metric = _get_metric(metric_string)
    device = images.device
    has_heatmap_bool = torch.tensor(has_heatmap, device=device).bool()

    # If no sample has a heatmap, return 0
    if not has_heatmap_bool.any():
        return (torch.tensor(0.0, device=device), None) if return_saliency \
               else torch.tensor(0.0, device=device)

    # Convert clickmaps to a tensor
    if isinstance(clickmaps, list):
        clickmaps = torch.stack(clickmaps, dim=0).to(device)

    # Build the one-hot for the true class
    batch_size, num_classes = outputs.shape
    one_hot = torch.zeros_like(outputs)
    one_hot.scatter_(1, labels.unsqueeze(1), 1.0)

    # Gradients wrt images -> saliency
    gradients = torch.autograd.grad(
        outputs=outputs,
        inputs=images,
        grad_outputs=one_hot,
        create_graph=True
    )[0]

    # Absolute saliency value then max across channel dimension
    saliency = gradients.abs().amax(dim=1, keepdim=True)  # shape [B, 1, H, W]

    # Apply Gaussian blur twice
    saliency = gaussian_blur(saliency, circle_kernel)
    saliency = gaussian_blur(saliency, circle_kernel)
    saliency_final = min_max_normalize_maps(saliency)

    # We now build gaussian pyramids
    saliency_pyramid = build_gaussian_pyramid(saliency_final, levels=pyramid_levels)
    clickmap_pyramid = build_gaussian_pyramid(clickmaps, levels=pyramid_levels)

    # Project the Saliency
    projected_saliency = model.module.saliency_projection(saliency_final)

    if get_top_k:
        # Compute top-k saliency per sample
        top_k_saliency = projected_saliency.clone()  # [B, 1, H, W]
        B, _, H, W = top_k_saliency.shape
        top_k_saliency = top_k_saliency.view(B, -1)  # [B, H*W]
        clickmaps_flat = clickmaps.view(B, -1)       # [B, H*W]
        for i in range(B):
            if has_heatmap_bool[i]:
                # Count non-zero pixels in the i-th clickmap as k_i
                k_i = (clickmaps_flat[i] > 0).sum().item()
                if k_i > 0:
                    sal_i = top_k_saliency[i]
                    top_vals, _ = torch.topk(sal_i, k_i)
                    threshold_i = top_vals[-1]
                    sal_i[sal_i < threshold_i] = 0.0
        projected_saliency = top_k_saliency.view(B, 1, H, W)

    # Normalize heatmaps
    clickmaps = min_max_normalize_maps(clickmaps)
    
    saliency_logits = projected_saliency.squeeze(1)

    # # Mask to only consider pixels where the raw blurred saliency > 0
    mask = (clickmaps.squeeze(1) > 0).float()
    
    # Prepare the target clickmap (also shape [B, H, W])
    if clickmaps.ndim == 4 and clickmaps.shape[1] == 1:
        target = clickmaps.squeeze(1)
    else:
        target = clickmaps
    
    # Compute BCE loss only on the valid pixels
    loss_per_sample = BCE_masked(saliency_logits, target, mask)
    harmonization_loss = loss_per_sample.mean()
    saliency_to_return = saliency_final.clone() if return_saliency else None

    if return_saliency:
        return harmonization_loss, saliency_to_return
    return harmonization_loss


    # --- Z-score normalization for saliency map per image ---
    # B, C, H, W = valid_saliency.shape
    # valid_saliency_flat = valid_saliency.view(B, -1)
    # mean = valid_saliency_flat.mean(dim=1, keepdim=True)
    # std = valid_saliency_flat.std(dim=1, keepdim=True) + 1e-7
    # z_valid_saliency = (valid_saliency_flat - mean) / std
    # z_valid_saliency = z_valid_saliency.view(B, C, H, W)
    # valid_saliency = z_valid_saliency


    # saliency_pyramid = build_gaussian_pyramid(saliency_norm, levels=pyramid_levels)
    # clickmap_pyramid = build_gaussian_pyramid(clickmaps_norm, levels=pyramid_levels)

    # TODO: Add the gaussian pyramid back in
    # Using Gaussian Pyramid
    # loss_levels = []
    # for level_idx in range(pyramid_levels):
    #     level_differences = metric(saliency_pyramid[level_idx], clickmap_pyramid[level_idx])
    #     level_loss = torch.sum(level_differences * has_heatmap_bool) / heatmap_count
    #     loss_levels.append(level_loss)
    # harmonization_loss = torch.mean(torch.stack(loss_levels))


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
