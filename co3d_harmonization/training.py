# training.py
import os
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


from .losses import *
from .utils import compute_spearman_correlation, apply_colormap, min_max_normalize_maps, denormalize_image, get_circle_kernel
from .config import WANDB_LOGGING, EPOCH_INTERVAL, KERNEL_SIZE, KERNEL_SIZE_SIGMA

circle_kernel = get_circle_kernel(KERNEL_SIZE, KERNEL_SIZE_SIGMA)

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
    # os.makedirs(plot_save_dir, exist_ok=True)

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
            model=model,
            images=images, 
            outputs=outputs, 
            labels=labels, 
            clickmaps=clickmaps, 
            has_heatmap=has_heatmap,
            pyramid_levels=pyramid_levels,
            return_saliency=True,
            metric_string=metric_string,
            get_top_k=True,
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

        if batch_idx % EPOCH_INTERVAL == 0:
            print(f"Batch {batch_idx}/{len(dataloader)}\tCE Loss: {ce_loss.item():.4f}\tHarm Loss: {harmonization_loss.item():.4f}\tTotal Loss: {total_loss.item():.4f}")

            # Visualize on WandB
            if WANDB_LOGGING:
                # Check if there is any valid heatmap in the batch
                has_heatmap_bool = torch.tensor(has_heatmap, device=device).bool()
                valid_indices = torch.nonzero(has_heatmap_bool, as_tuple=False).flatten()
                if len(valid_indices) > 0:
                    # Pick the first valid index
                    sample_idx = valid_indices[0].item()
                else:
                    # If no valid heatmaps, default to sample 0
                    sample_idx = 0
                
                v_image = images[sample_idx].detach().cpu()
                v_clickmap = clickmaps[sample_idx].detach().cpu()

                # The saliency map is already normalized, but the clickmap is not
                # so we min-max normalize only the clickmap
                v_clickmap = min_max_normalize_maps(v_clickmap)

                if saliency_map is not None:
                    v_saliency = saliency_map[sample_idx].detach().cpu()
                else:
                    v_saliency = torch.zeros_like(v_clickmap)

                # If the image has shape [3, H, W], transpose to [H, W, 3]
                if v_image.ndim == 3 and v_image.shape[0] in [1, 3]:
                    v_image = v_image.permute(1, 2, 0)  # [C, H, W] -> [H, W, C]

                # If the clickmap and saliency has shape [1, H, W] we change it to [H, W]
                if v_clickmap.ndim == 3 and v_clickmap.shape[0] == 1:
                    v_clickmap = v_clickmap.squeeze(0)
                if v_saliency.ndim == 3 and v_saliency.shape[0] == 1:
                    v_saliency = v_saliency.squeeze(0)

                # Convert those 2D arrays to color images via apply_colormap
                colored_clickmap = apply_colormap(v_clickmap, cmap="viridis")
                colored_saliency = apply_colormap(v_saliency, cmap="viridis")

                # Convert the original image to [H,W,3] uint8
                v_image = denormalize_image(v_image) 
                # Now convert to uint8
                v_image_np = (v_image.numpy() * 255).astype(np.uint8)

                # Stacking all 3 together for visualizations
                stacked_visualization = np.concatenate([v_image_np, colored_clickmap, colored_saliency], axis=1)

                # Also printing correlation for reference
                current_correlation = compute_spearman_correlation(v_saliency, v_clickmap)

                # wandb Image
                log_dict = {"training_visualization": wandb.Image(stacked_visualization, caption=f"(Image | Clickmap | Saliency) Correlation: {current_correlation:.4f}")}

                wandb.log(log_dict)

    avg_ce_loss = running_ce_loss / len(dataloader)
    avg_harm_loss = running_harm_loss / len(dataloader)
    avg_total_loss = running_total_loss / len(dataloader)
    accuracy = running_corrects / total_samples
    print(f"Epoch {epoch} Average Training Loss: {avg_total_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    if WANDB_LOGGING:
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

        # Using the harmonization loss function to compute get only the saliency maps
        # we don't care about the loss here, only the saliency maps
        _, saliency_map = compute_harmonization_loss(
            model=model,
            images=images, 
            outputs=outputs, 
            labels=labels, 
            clickmaps=clickmaps, 
            has_heatmap=has_heatmap,
            pyramid_levels=pyramid_levels,
            return_saliency=True,
            metric_string="BCE",
            get_top_k=True,
        )

        # Processing the clickmaps and saliency maps to compute alignment
        # saliency is already preprocessed in the harmonization loss function
        clickmaps = min_max_normalize_maps(clickmaps)

    
        # Compute alignment here after processing the clickmaps and the saliency maps
        valid_indices = torch.nonzero(has_heatmap_bool).squeeze(1)
        for idx in valid_indices:
            sample_saliency = saliency_map[idx:idx+1]      
            sample_clickmap = clickmaps[idx:idx+1]
            alignment = compute_spearman_correlation(sample_saliency, sample_clickmap)
            alignment_scores.append(alignment)
        
        # Validation Visualization: Log only for the first batch.
        if WANDB_LOGGING:
            # Check if there is any valid heatmap in the batch
            has_heatmap_bool = torch.tensor(has_heatmap, device=device).bool()
            valid_indices = torch.nonzero(has_heatmap_bool, as_tuple=False).flatten()
            if len(valid_indices) > 0:
                # Pick the first valid index
                sample_idx = valid_indices[0].item()
            else:
                # If no valid heatmaps, default to sample 0
                sample_idx = 0
            
            v_image = images[sample_idx].detach().cpu()
            v_clickmap = clickmaps[sample_idx].detach().cpu()

            if saliency_map is not None:
                v_saliency = saliency_map[sample_idx].detach().cpu()
            else:
                v_saliency = torch.zeros_like(v_clickmap)

            # If the image has shape [3, H, W], transpose to [H, W, 3]
            if v_image.ndim == 3 and v_image.shape[0] in [1, 3]:
                v_image = v_image.permute(1, 2, 0)  # [C, H, W] -> [H, W, C]

            # If the clickmap and saliency has shape [1, H, W] we change it to [H, W]
            if v_clickmap.ndim == 3 and v_clickmap.shape[0] == 1:
                v_clickmap = v_clickmap.squeeze(0)
            if v_saliency.ndim == 3 and v_saliency.shape[0] == 1:
                v_saliency = v_saliency.squeeze(0)

            # Convert those 2D arrays to color images via apply_colormap
            colored_clickmap = apply_colormap(v_clickmap, cmap="viridis")
            colored_saliency = apply_colormap(v_saliency, cmap="viridis")

            # Convert the original image to [H,W,3] uint8
            v_image = denormalize_image(v_image) 
            # Now convert to uint8
            v_image_np = (v_image.numpy() * 255).astype(np.uint8)

            # Stacking all 3 together for visualizations
            stacked_visualization = np.concatenate([v_image_np, colored_clickmap, colored_saliency], axis=1)

            # Also printing correlation for reference
            current_correlation = compute_spearman_correlation(v_saliency, v_clickmap)

            # wandb Image
            log_dict = {"validation_visualization": wandb.Image(stacked_visualization, caption=f"(Image | Clickmap | Saliency) Correlation: {current_correlation:.4f}")}

    avg_ce_loss = running_ce_loss / len(dataloader)
    accuracy = running_corrects / total_samples
    avg_alignment = (sum(alignment_scores) / len(alignment_scores)) if alignment_scores else 0.0

    return avg_ce_loss, accuracy, avg_alignment