"""
Harmonization loss module for aligning model saliency maps with human attention maps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyramid import pyramidal_representation
from utils import compute_human_alignment

# def standardize_cut(heatmaps, axis=(2, 3), epsilon=1e-5):
#     """
#     Standardize the heatmaps (zero mean, unit variance) and apply ReLU.

#     Parameters:
#     -----------
#     heatmaps : torch.Tensor
#         The heatmaps to standardize. Shape: (N, 1, H, W)
#     axis : tuple, optional
#         The axes to compute the mean and variance. Default: (2, 3)
#     epsilon : float, optional
#         A small value to avoid division by zero. Default: 1e-5

#     Returns:
#     --------
#     torch.Tensor
#         The positive part of the standardized heatmaps.
#     """
#     means = torch.mean(heatmaps, dim=axis, keepdim=True)
#     stds = torch.std(heatmaps, dim=axis, keepdim=True)
#     heatmaps = (heatmaps - means) / (stds + epsilon)
#     return torch.relu(heatmaps)

def mse(heatmaps_a, heatmaps_b):
    """
    Compute the Mean Squared Error between two sets of heatmaps.

    Parameters:
    -----------
    heatmaps_a : torch.Tensor
        The first set of heatmaps.
    heatmaps_b : torch.Tensor
        The second set of heatmaps.

    Returns:
    --------
    torch.Tensor
        The Mean Squared Error.
    """
    return torch.mean(torch.square(heatmaps_a - heatmaps_b))

def pyramidal_mse(true_heatmaps, predicted_heatmaps, nb_levels=5):
    """
    Compute mean squared error between two sets of heatmaps on a pyramidal representation.

    Parameters:
    -----------
    true_heatmaps : torch.Tensor
        The true heatmaps. Shape: (N, 1, H, W)
    predicted_heatmaps : torch.Tensor
        The predicted heatmaps. Shape: (N, 1, H, W)
    nb_levels : int, optional
        The number of levels to use in the pyramid. Default: 5

    Returns:
    --------
    torch.Tensor
        The weighted MSE across all pyramid levels.
    """
    # print("Shape of heatmaps:", true_heatmaps.shape, predicted_heatmaps.shape)
    pyramid_y = pyramidal_representation(true_heatmaps, nb_levels)
    pyramid_y_pred = pyramidal_representation(predicted_heatmaps, nb_levels)
    
    loss = torch.mean(torch.stack(
        [mse(pyramid_y[i], pyramid_y_pred[i]) for i in range(nb_levels)]))
    return loss

def harmonizer_loss(predicted_label, true_label, saliency_maps, clickme_maps, lambda_weights=1e-5, lambda_harmonization=1.0):
    """
    Compute the harmonization loss: cross entropy + pyramidal MSE of standardized-cut heatmaps.

    Parameters:
    -----------
    model : torch.nn.Module
        The model to train.
    images : torch.Tensor
        The batch of images to train on.
    labels : torch.Tensor
        The batch of labels.
    clickme_maps : torch.Tensor
        The batch of true heatmaps (e.g., Click-me maps) to align the model on.
    criterion : torch.nn.CrossEntropyLoss
        The cross entropy loss to use.
    lambda_weights : float, optional
        The weight decay factor. Default: 1e-5
    lambda_harmonization : float, optional
        The weight for the harmonization loss. Default: 1.0

    Returns:
    --------
    harmonization_loss : torch.Tensor
        The total harmonization loss.
    cce_loss : torch.Tensor
        The cross-entropy loss component.
    """
    # Standardize and normalize heatmaps
    # saliency_maps = standardize_cut(saliency_maps)
    # clickme_maps = standardize_cut(clickme_maps)
    
    # saliency_max = torch.amax(saliency_maps, (2,3), keepdims=True) + 1e-6
    # clickme_max = torch.amax(clickme_maps, (2,3), keepdims=True) + 1e-6
    # clickme_maps = clickme_maps / clickme_max * saliency_max
    
    # Compute losses
    pyramidal_loss = pyramidal_mse(saliency_maps, clickme_maps)
    cce_loss = nn.CrossEntropyLoss()(predicted_label, true_label)
    harmonization_loss = pyramidal_loss * lambda_harmonization

    return harmonization_loss, cce_loss

def harmonization_eval(model, images, labels, clickme_maps, criterion):
    """
    Evaluate the model's harmonization performance.

    Parameters:
    -----------
    model : torch.nn.Module
        The model to evaluate.
    images : torch.Tensor
        The batch of images to evaluate on.
    labels : torch.Tensor
        The batch of labels.
    clickme_maps : torch.Tensor
        The batch of true heatmaps (e.g., Click-me maps) to compare against.
    criterion : torch.nn.CrossEntropyLoss
        The cross entropy loss to use.

    Returns:
    --------
    output : torch.Tensor
        The model's output predictions.
    cce_loss : torch.Tensor
        The cross-entropy loss.
    human_alignment : torch.Tensor
        The score indicating the Spearman correlation between saliency maps and clickme maps.
    """
    model.eval()
    images.requires_grad = True
    output = model(images)
    cce_loss = criterion(output, labels)
    
    # Compute saliency maps
    correct_class_scores = output.gather(1, labels.view(-1, 1)).squeeze()
    device = images.device
    ones_tensor = torch.ones(correct_class_scores.shape).to(device)
    correct_class_scores.backward(ones_tensor, retain_graph=True)
    
    grads = torch.abs(images.grad)
    saliency_maps, _ = torch.max(grads, dim=1, keepdim=True) # saliency map (N, C, H, W) -> (N, 1, H, W)
    
    # measure human alignment
    human_alignment = compute_human_alignment(saliency_maps, clickme_maps)
    images.grad.zero_()
    
    return output, cce_loss, human_alignment

if __name__ == "__main__":
    # Test code
    hmp_a, hmp_b = torch.rand(4, 1, 224, 224), torch.rand(4, 1, 224, 224)
    print(standardize_cut(hmp_a).shape)
    print(mse(hmp_a, hmp_b))
    print(pyramidal_mse(hmp_a, hmp_b))