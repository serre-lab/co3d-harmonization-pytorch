# training.py
import os
import wandb
import torch
import torch.nn as nn
import torch.optim as optim

from .losses import *
from .utils import compute_spearman_correlation
from .config import WANDB_LOGGING

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
            images=images, 
            outputs=outputs, 
            labels=labels, 
            clickmaps=clickmaps, 
            has_heatmap=has_heatmap, 
            pyramid_levels=pyramid_levels,
            return_saliency=True,
            metric_string=metric_string
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

        # Visualize and save the plot only for a sample with a valid heatmap.
        # sample_to_plot = None
        # for i, flag in enumerate(has_heatmap):
        #     if flag:
        #         sample_to_plot = i
        #         break

        # if sample_to_plot is not None:
        #     fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        #     # Display the image.
        #     axs[0].imshow(images[sample_to_plot].detach().cpu().permute(1, 2, 0))
        #     axs[0].set_title("Image")
        #     # Display the saliency map.
        #     if saliency_map is not None:
        #         axs[1].imshow(saliency_map[sample_to_plot].detach().cpu().squeeze(0), cmap='viridis')
        #         axs[1].set_title("Saliency Map")
        #     else:
        #         axs[1].text(0.5, 0.5, "No Saliency", horizontalalignment='center')
        #         axs[1].set_title("Saliency Map")
        #     # Display the clickmap.
        #     axs[2].imshow(clickmaps[sample_to_plot].detach().cpu().squeeze(0), cmap='gray')
        #     axs[2].set_title("Clickmap")
        #     # Save the figure.
        #     plot_filename = os.path.join(plot_save_dir, f"epoch{epoch}_batch{batch_idx}_sample{sample_to_plot}.png")
        #     plt.savefig(plot_filename)
        #     plt.close(fig)

        if batch_idx % 10 == 0:
            print(
                f"Epoch {epoch} Batch {batch_idx}: "
                f"CE Loss: {ce_loss.item():.4f}, "
                f"Harm Loss: {harmonization_loss.item():.4f}, "
                f"Total Loss: {total_loss.item():.4f}"
            )

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

        one_hot = torch.zeros_like(outputs)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)

        gradients = torch.autograd.grad(
            outputs=outputs,
            inputs=images,
            grad_outputs=one_hot,
            retain_graph=True,
            allow_unused=True
        )[0]

        saliency = gradients.abs().amax(dim=1, keepdim=True)

        valid_indices = torch.nonzero(has_heatmap_bool).squeeze(1)
        for idx in valid_indices:
            sample_saliency = saliency[idx:idx+1]      
            sample_clickmap = clickmaps[idx:idx+1]       
            alignment = compute_spearman_correlation(sample_saliency, sample_clickmap)
            alignment_scores.append(alignment)

    avg_ce_loss = running_ce_loss / len(dataloader)
    accuracy = running_corrects / total_samples
    avg_alignment = (sum(alignment_scores) / len(alignment_scores)) if alignment_scores else 0.0

    print(f"Validation CCE Loss: {avg_ce_loss:.4f}, Accuracy: {accuracy:.4f}, Alignment Score: {avg_alignment:.4f}")
    if WANDB_LOGGING:
        wandb.log({
            "val_avg_ce_loss": avg_ce_loss, 
            "val_accuracy": accuracy, 
            "val_alignment_score": avg_alignment,
        })
    return avg_ce_loss, accuracy, avg_alignment