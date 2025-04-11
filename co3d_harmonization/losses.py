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
        return BCE
    elif metric_string == "maskedBCE":
        return BCE_masked
    else:
        raise ValueError("Invalid metric. Must be 'CE', 'MSE', 'cosine', or 'BCE'.")

def compute_harmonization_loss(model, images, outputs, labels, clickmaps, has_heatmap, pyramid_levels=5, metric_string="CE", return_saliency=False, get_top_k=False):
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



def compute_FEL_harmonization_loss(model, images, outputs, labels, clickmaps, has_heatmap, pyramid_levels=5, metric_string="CE", return_saliency=False, get_top_k=False):
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

    # No need to apply Gaussian blur becuase the pyramid will do that
    saliency_final = min_max_normalize_maps(saliency)

    # We now build gaussian pyramids
    saliency_pyramid = build_gaussian_pyramid(saliency_final, levels=pyramid_levels)
    clickmap_pyramid = build_gaussian_pyramid(clickmaps, levels=pyramid_levels)

    # Iterate through the levels of the pyramid and compute MSE loss at each layer
    loss_levels = []
    for level_idx in range(pyramid_levels):
        level_differences = metric(saliency_pyramid[level_idx], clickmap_pyramid[level_idx])
        level_loss = torch.sum(level_differences * has_heatmap_bool) / has_heatmap_bool.sum()
        loss_levels.append(level_loss)

    # Compute the final harmonization loss as the mean of the losses across all levels
    harmonization_loss = torch.mean(torch.stack(loss_levels))

    if return_saliency:
        return harmonization_loss, saliency_final.clone()
    return harmonization_loss


import torch
import torch.nn.functional as F

def compute_contrastive_harmonization_loss(
    images,
    outputs,
    labels,
    clickmaps,
    has_heatmap,
    logit_scale_value=1.0,   
    return_saliency=False
):
    """
    Contrastive harmonization loss inspired by CLIP:
      1. Compute gradient-based saliency maps for each image
      2. Interpret each saliency map and its clickmap as two embeddings
      3. Flatten & L2-normalise
      4. Compute pairwise dot products across the batch and convert them to logits
      5. CE loss that matches each saliency with its own clickmap and vice versa

    Args:
        images (torch.Tensor): [B, C, H, W], requires_grad=True
        outputs (torch.Tensor): [B, num_classes] (model logits)
        labels (torch.Tensor): [B]
        clickmaps (torch.Tensor or list): [B, 1, H, W] heatmaps or a list
        has_heatmap (list or tensor): Boolean indicator per sample
        logit_scale_value (float): A scalar factor for the logits (like CLIP's logit_scale.exp()).
        return_saliency (bool): If True, also return the saliency maps.

    Returns:
        torch.Tensor or (torch.Tensor, torch.Tensor):
            contrastive_loss, or (contrastive_loss, saliency_map)
    """
    device = images.device

    has_heatmap_bool = torch.tensor(has_heatmap, device=device).bool()
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

    saliency = gradients.abs().amax(dim=1, keepdim=True)

    saliency_norm = min_max_normalize_maps(saliency)     # shape [B,1,H,W]
    clickmaps_norm = min_max_normalize_maps(clickmaps)   # shape [B,1,H,W]

    B, _, H, W = saliency_norm.shape
    saliency_flat = saliency_norm.view(B, -1)
    clickmap_flat = clickmaps_norm.view(B, -1)

    saliency_emb = F.normalize(saliency_flat, p=2, dim=1)
    clickmap_emb = F.normalize(clickmap_flat, p=2, dim=1)

    # Same as the CLIP line: image_features @ text_features.T
    logits_per_saliency = logit_scale_value * (saliency_emb @ clickmap_emb.t())
    logits_per_clickmap = logit_scale_value * (clickmap_emb @ saliency_emb.t())

    # Match each sample only with itself
    targets = torch.arange(B, device=device)

    # Cross-entropy in each direction and divide by 2 to average
    loss_s = F.cross_entropy(logits_per_saliency, targets)  # Saliency -> Clickmap
    loss_c = F.cross_entropy(logits_per_clickmap, targets)  # Clickmap -> Saliency
    contrastive_loss = (loss_s + loss_c) / 2.0

    saliency_to_return = saliency_norm.clone() if return_saliency else None
    if return_saliency:
        return contrastive_loss, saliency_to_return
    return contrastive_loss
