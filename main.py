import argparse
import torch
from torch.utils.data import DataLoader
import timm
import wandb
from torch import optim
import numpy as np
import random

from co3d_harmonization.dataset import ClickMe, collate_fn
from co3d_harmonization.training import train_one_epoch, validate
from co3d_harmonization.config import N_CO3D_CLASSES, WANDB_LOGGING, WANDB_USERNAME, WANDB_PROJECT

if WANDB_LOGGING:
    wandb.init(entity=WANDB_USERNAME, project=WANDB_PROJECT)

def main():
    # Setting seed for reproducibility
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)
    random.seed(0)

    # Parse arguments
    parser = argparse.ArgumentParser(description='Harmonized Training with ClickMe 2.0')
    parser.add_argument('--epochs', type=int, default=10, help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--train-folder', type=str, default="data/CO3D_ClickMe_Training/", help='training image folder')
    parser.add_argument('--val-folder', type=str, default="data/CO3D_ClickMe_Validation/", help='validation image folder')
    parser.add_argument('--lambda_value', type=float, default=1.0, help='harmonization loss weight')
    parser.add_argument('--ce_multiplier', type=float, default=1.0, help='multiplier for the CE component of the loss')
    parser.add_argument('--metric', type=str, default="cosine", help='metric to compute harmonization loss (CE, MSE, cosine, BCE)')
    parser.add_argument('--model', type=str, default="vit_small_patch16_224.augreg_in21k_ft_in1k", help='TIMM model to use')
    # parser.add_argument('--pretrained', action='strore_true', default=True, help='TIMM model to use')
    
    args = parser.parse_args()
    
    # Log training device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print("Training using GPU:", gpu_name)
    else:
        print("Training using CPU!")

    # Initialize datasets and dataloaders
    label_to_category_map = {}

    # Load the training and validation datasets
    train_dataset = ClickMe(args.train_folder, label_to_category_map, is_training=True)
    val_dataset = ClickMe(args.val_folder, label_to_category_map, is_training=False)
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Create model using timm
    model = timm.create_model(args.model, pretrained=True, num_classes=N_CO3D_CLASSES)
    model = torch.nn.DataParallel(model).to(device)

    # SGD Optimizer with momentum; exactly as it is in the paper
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    # Baseline validation pass
    print("Baseline Validation:")
    baseline_loss, baseline_acc, baseline_alignment = validate(model, val_dataloader, device)
    print(f"Baseline Val Loss: {baseline_loss:.4f}, Acc: {baseline_acc:.4f}, Align: {baseline_alignment:.4f}")
    if WANDB_LOGGING:
        wandb.log({
            "val_avg_ce_loss": baseline_loss,
            "val_accuracy": baseline_acc,
            "val_alignment_score": baseline_alignment,
            "epoch": 0
        })

    # Training and validation loop
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}:")
        # Train for one epoch has the logging within the function
        train_one_epoch(model, train_dataloader, optimizer, device, epoch, args.lambda_value, args.ce_multiplier, args.metric)
        val_loss, val_acc, val_alignment = validate(model, val_dataloader, device)

        # Logging the validaton results
        print(f"Validation CCE Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, Alignment Score: {val_alignment:.4f}\n")
        
        if WANDB_LOGGING:
            wandb.log({
                "val_avg_ce_loss": val_loss, 
                "val_accuracy": val_acc, 
                "val_alignment_score": val_alignment,
                "epoch": epoch
            })

    # Finish wandb logging
    wandb.finish()

if __name__ == '__main__':
    main()