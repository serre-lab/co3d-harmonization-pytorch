import torch
import torch.nn.functional as F
import time

"""
Reference: 
-   PyTorch Implementation of Laplacian Pyramid Loss
    https://gist.github.com/Harimus/918fddd8bdc6e13e4acf3d213f2f24cd
"""

def downsample(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    Downsample an image using a given kernel.

    Args:
        image (torch.Tensor): Input image tensor of shape (batch, channels, height, width).
        kernel (torch.Tensor): Downsampling kernel.

    Returns:
        torch.Tensor: Downsampled image.

    Raises:
        IndexError: If the input tensor shape is not (batch, channels, height, width).
    """
    if len(image.shape) != 4:
        raise IndexError(f'Expected input tensor to be of shape: (batch, channels, height, width) but got: {image.shape}')
    
    groups = kernel.shape[1]
    padding = kernel.shape[-1] // 2  # Kernel size needs to be odd number
    
    return F.conv2d(image, weight=kernel, stride=2, padding=padding, groups=groups)

def gaussian_kernel(num_channels: int) -> torch.Tensor:
    """
    Create a Gaussian kernel for the given number of channels.

    Args:
        num_channels (int): Number of channels in the input image.

    Returns:
        torch.Tensor: Gaussian kernel of shape (num_channels, 1, 5, 5).
    """
    kernel = torch.tensor([
        [1., 4., 7., 4., 1],
        [4., 16., 26., 16., 4.],
        [7., 26., 41., 26., 7.],
        [4., 16., 26., 16., 4.],
        [1., 4., 6., 4., 1.]
    ]) / 256.0  # 5x5 Gaussian Kernel

    return kernel.repeat(num_channels, 1, 1, 1)  # (C, 1, H, W)

def pyramidal_representation(image: torch.Tensor, num_levels: int) -> list[torch.Tensor]:
    """
    Compute the pyramidal representation of an image.

    Args:
        image (torch.Tensor): Input image tensor of shape (batch, channels, height, width).
        num_levels (int): The number of levels to use in the pyramid.

    Returns:
        list[torch.Tensor]: The pyramidal representation as a list of tensors.
    """
    device = image.device
    kernel = gaussian_kernel(image.shape[1]).to(device)
    levels = [image]
    for _ in range(num_levels - 1):
        image = downsample(image, kernel)
        levels.append(image)
    return levels

def benchmark_downsampling(image: torch.Tensor, num_levels: int) -> None:
    """
    Benchmark the downsampling process and print the results.

    Args:
        image (torch.Tensor): Input image tensor of shape (batch, channels, height, width).
        num_levels (int): The number of levels to use in the pyramid.
    """
    kernel = gaussian_kernel(image.shape[1])
    padding = kernel.shape[-1] // 2
    
    start = time.time()
    for _ in range(num_levels):
        mask = torch.ones(1, *image.shape[1:])
        mask = F.conv2d(mask, kernel, groups=image.shape[1], stride=2, padding=padding)
        image = F.conv2d(image, kernel, groups=image.shape[1], stride=2, padding=padding)
        image = image / mask  # Normalize the edges and corners.
        print(image.shape)
    end = time.time()
    print(f"Time taken: {end - start:.4f} seconds")

if __name__ == '__main__':
    # Test pyramidal representation
    img = torch.rand(64, 1, 128, 128)
    reps = pyramidal_representation(img, 5)
    print("Pyramidal representation shapes:")
    for rep in reps:
        print(rep.shape)
    
    print("\nBenchmarking downsampling:")
    benchmark_img = torch.rand(1, 3, 224, 224)
    benchmark_downsampling(benchmark_img, 5)