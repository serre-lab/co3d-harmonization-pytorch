import argparse
import torch
from torch.utils.data import DataLoader
import timm
import wandb
from torch import optim

from co3d_harmonization.dataset import ClickMe, collate_fn
from co3d_harmonization.training import train_one_epoch, validate
from co3d_harmonization.config import N_CO3D_CLASSES, WANDB_LOGGING

if WANDB_LOGGING:
    wandb.init(entity="grassknoted", project="co3d-harmonization")

def main():
    parser = argparse.ArgumentParser(description='Harmonized Training with ClickMe 2.0')
    parser.add_argument('--epochs', type=int, default=10, help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--train-folder', type=str, default="data/CO3D_ClickMe_Training/", help='training image folder')
    parser.add_argument('--val-folder', type=str, default="data/CO3D_ClickMe_Validation/", help='validation image folder')
    parser.add_argument('--lambda_value', type=float, default=1.0, help='harmonization loss weight')
    parser.add_argument('--ce_multiplier', type=float, default=1.0, help='multiplier for the CE component of the loss')
    parser.add_argument('--metric', type=str, default="cosine", help='metric to compute harmonization loss (CE, MSE, cosine)')
    parser.add_argument('--model', type=str, default="vit_small_patch16_224.augreg_in21k_ft_in1k", help='TIMM model to use')
    # parser.add_argument('--pretrained', action='strore_true', default=True, help='TIMM model to use')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # Initialize datasets and dataloaders
    label_to_category_map = {}
    train_dataset = ClickMe(args.train_folder, label_to_category_map, is_training=True)
    val_dataset = ClickMe(args.val_folder, label_to_category_map, is_training=False)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Create model using timm
    model = timm.create_model(args.model, pretrained=True, num_classes=N_CO3D_CLASSES)
    model = torch.nn.DataParallel(model).to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    for epoch in range(1, args.epochs + 1):
        print(f"Starting Epoch {epoch}")
        train_one_epoch(model, train_dataloader, optimizer, device, epoch, args.lambda_value, args.ce_multiplier, args.metric)
        validate(model, val_dataloader, device)

    wandb.finish()

if __name__ == '__main__':
    main()