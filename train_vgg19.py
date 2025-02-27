#!/usr/bin/env python3
"""
Train VGG-19 on ImageNet using TIMM with multi-GPU support, TQDM progress bars,
label smoothing, mixup augmentation, and a custom LR scheduler.

Usage:
    CUDA_VISIBLE_DEVICES=0,1 python train_vgg19_imagenet.py \
         --data /path/to/imagenet --epochs 90 --batch-size 256
"""

import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms

import timm
from tqdm import tqdm


##############################################
# Label Smoothing Cross-Entropy Loss
##############################################

class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        # pred: (B, num_classes)
        # target: (B,) with class indices
        num_classes = pred.size(1)
        log_probs = torch.log_softmax(pred, dim=1)
        
        # Create one-hot encoding with smoothing
        with torch.no_grad():
            true_dist = torch.empty_like(log_probs)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=1))


##############################################
# Mixup Augmentation
##############################################

def mixup_data(x, y, alpha=0.2):
    """
    Returns mixed inputs, pairs of targets, and lambda.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


##############################################
# Training and Validation Routines
##############################################

def train_one_epoch(train_loader, model, criterion, optimizer, device, epoch, mixup_alpha):
    model.train()
    running_loss = 0.0
    total_samples = 0
    start_time = time.time()

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} Training", leave=False)
    for inputs, targets in progress_bar:
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Apply mixup augmentation if enabled.
        if mixup_alpha > 0:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=mixup_alpha)
            outputs = model(inputs)
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)
        progress_bar.set_postfix(loss=loss.item())

    epoch_loss = running_loss / total_samples
    elapsed_time = time.time() - start_time
    print(f"Epoch {epoch} Training Loss: {epoch_loss:.4f} | Time: {elapsed_time:.0f}s")
    return epoch_loss


def validate(val_loader, model, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    total_samples = 0
    running_acc = 0.0

    progress_bar = tqdm(val_loader, desc=f"Epoch {epoch} Validation", leave=False)
    with torch.no_grad():
        for inputs, targets in progress_bar:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            acc1, _ = accuracy(outputs, targets, topk=(1, 5))
            running_acc += acc1.item() * inputs.size(0) / 100.0
            total_samples += inputs.size(0)
            progress_bar.set_postfix(loss=loss.item())

    epoch_loss = running_loss / total_samples
    epoch_acc = running_acc / total_samples * 100.0
    print(f"Epoch {epoch} Validation Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")
    return epoch_loss, epoch_acc


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


##############################################
# Main
##############################################

def main():
    parser = argparse.ArgumentParser(
        description='Train VGG-19 on ImageNet using TIMM with multi-GPU, label smoothing, mixup, '
                    'and a custom LR scheduler.'
    )
    parser.add_argument('--data', default='/path/to/imagenet', type=str,
                        help='Path to ImageNet dataset (expects subfolders "train" and "val")')
    parser.add_argument('--epochs', default=90, type=int,
                        help='Number of total epochs to run')
    parser.add_argument('--batch-size', default=256, type=int,
                        help='Mini-batch size')
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='Momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='Weight decay')
    parser.add_argument('--num-workers', default=8, type=int,
                        help='Number of data loading workers')
    parser.add_argument('--model', default='vgg19', type=str,
                        help='Model name from TIMM (default: vgg19)')
    parser.add_argument('--save-path', default='vgg19_imagenet.pth', type=str,
                        help='Path to save the best model')
    parser.add_argument('--label-smoothing', default=0.1, type=float,
                        help='Label smoothing factor')
    parser.add_argument('--mixup-alpha', default=0.2, type=float,
                        help='Mixup alpha parameter (0 to disable mixup)')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_dir = os.path.join(args.data, 'train')
    val_dir = os.path.join(args.data, 'val')

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    print(f'Creating model {args.model}')
    model = timm.create_model(args.model, pretrained=False, num_classes=1000)
    model = model.to(device)

    # Multi-GPU support.
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        print(f"Multiple GPUs detected ({torch.cuda.device_count()}). Using DataParallel.")
        model = nn.DataParallel(model)

    criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing).to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)

    def lr_lambda(epoch):
        warmup_epochs = 5
        total_epochs = args.epochs
        if epoch < warmup_epochs:
            lr_mult = float(epoch) / warmup_epochs
        else:
            lr_mult = 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))
        # Additional decays:
        if epoch >= 30:
            lr_mult *= 0.1
        if epoch >= 50:
            lr_mult *= 0.1
        if epoch >= 80:
            lr_mult *= 0.1
        return lr_mult

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_one_epoch(train_loader, model, criterion, optimizer, device, epoch, args.mixup_alpha)
        _, val_acc = validate(val_loader, model, criterion, device, epoch)
        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            print(f"New best accuracy: {best_acc:.2f}%. Saving model...")
            torch.save(model.state_dict(), args.save_path)

    print("Training complete.")


if __name__ == '__main__':
    main()
