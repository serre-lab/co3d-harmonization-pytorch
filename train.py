import argparse
import torch
import math
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torch.utils.data import DataLoader
from dataset import ClickMe
from loss import harmonizer_loss
import utils
import numpy as np
from tqdm import tqdm
from scipy.stats import spearmanr
from torchmetrics.functional import spearman_corrcoef
import timm
import matplotlib.pyplot as plt
import json
import torchvision.transforms.functional as tvF
import torch.nn.functional as F
import os

N_CO3D_CLASSES = 51
BRUSH_SIZE = 11
BRUSH_SIZE_SIGMA = np.sqrt(BRUSH_SIZE)
HUMAN_SPEARMAN_CEILING = 0.4422303328731989

# Add wandb logging
import wandb    

wandb.init(entity="grassknoted", project="co3d-harmonization")

def parse_args():
    parser = argparse.ArgumentParser(description="Train a ViT model on ClickMeDataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training (cuda/cpu)")
    parser.add_argument("--output_model", type=str, default="trained_vit_clickme.pth",
                        help="Path to save the trained model")
    return parser.parse_args()

def main(args):
    # Move the model to the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    label_to_category_map = {}

    # Load ClickMeDataset
    train_dataset = ClickMe(image_folder="data/CO3D_ClickMe2/", csv_path="data/clickme_vCO3D.csv", label_to_category_map=label_to_category_map, is_training=True)
    val_dataset = ClickMe(image_folder="data/CO3D_ClickMe2/", csv_path="data/clickme_vCO3D.csv", label_to_category_map=label_to_category_map, is_training=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Load the ViT small model
    model = timm.create_model('vit_small_patch16_224', pretrained=True)
    model.heads = nn.Linear(model.embed_dim, N_CO3D_CLASSES)
    model = model.to(device)

    # Define optimizer and custom loss
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    

    gaussian_kernel = utils.gaussian_kernel(size=BRUSH_SIZE, sigma=BRUSH_SIZE_SIGMA)

    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        train_correct = 0
        total_samples = 0

        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{args.num_epochs}"):
            images, heatmaps, targets = batch[0], batch[1], batch[2]
            images, heatmaps, targets = images.to(device), heatmaps.to(device), targets.to(device)

            images.requires_grad = True

            optimizer.zero_grad()
            outputs = model(images)
            correct_class_scores = outputs.gather(1, targets.view(-1, 1)).squeeze()
            device = images.device
            ones_tensor = torch.ones(correct_class_scores.shape).to(device) # scores is a tensor here, need to supply initial gradients of same tensor shape as scores.

            # obtain saliency map
            grads = torch.autograd.grad(outputs=correct_class_scores, inputs=images, grad_outputs=ones_tensor, retain_graph=True, create_graph=True, only_inputs=True)[0]
            saliency_maps = torch.mean(grads, dim=1, keepdim=True)
            heatmaps = utils.gaussian_blur(heatmaps.unsqueeze(1), gaussian_kernel).unsqueeze(1)

            loss = harmonizer_loss(outputs, targets, heatmaps, saliency_maps)[0] # 0 -> harmonization loss, 1 -> cross entropy loss
            loss.backward()
            optimizer.step()

            train_correct += (torch.argmax(outputs, dim=1) == targets).sum().item()
            total_samples += targets.size(0) 
    
        train_acc = train_correct / total_samples
         
        print(f"Epoch [{epoch+1}/{args.num_epochs}], Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}")
        # eval_alignment(model, val_loader, device, epoch, None, list(range(10)), args)
            #   f"Val Loss: {val_loss/len(val_loader):.4f}")

    print(f"Training completed. Model saved to {args.output_model}")

def eval_alignment(full_model:torch.nn.Module, test_data_loader, device: torch.device, epoch: int, log_writer, visualize, args):

    sample_maps = []
    criterion = nn.CrossEntropyLoss()
    alignment_scores = []
    null_scores = []
    all_test_acc = []
    category_alignments = {}
    category_nulls = {}
    medians = json.load(open("data/click_medians.json", "r"))

    # Set the model to eval mode
    full_model.eval()

    kernel = utils.gaussian_kernel(size=BRUSH_SIZE, sigma=math.sqrt(BRUSH_SIZE)).to(device)
    for i, batch in tqdm(enumerate(test_data_loader)):
        imgs, hmps, labels, img_names, cat = batch
        imgs, hmps, labels = imgs.to(device), hmps.to(device), labels.to(device)
        img_name = img_names[0]
        cat = cat[0]
        sub_vec = np.where(test_data_loader.dataset.categories != cat)[0]
        random_idx = np.random.choice(sub_vec)
        random_hmps = test_data_loader.dataset[random_idx]
        random_hmps = torch.unsqueeze(torch.Tensor(random_hmps[1]), 0)

        if cat not in category_alignments.keys():
            category_alignments[cat] = []
            category_nulls[cat] = []
        img_name = img_name.replace(f'{cat}_', f'{cat}/')
        if len(visualize)>0 and i in visualize:
            img = imgs.clone().detach().cpu().numpy().squeeze()
            img = np.moveaxis(img, 0, -1)
            # img = img*args.std + args.mean
            img = img*full_model.std + full_model.mean
            img = np.uint8(255*img)
        imgs.requires_grad = True
        outputs = full_model(imgs)
        loss = criterion(outputs, labels)

        test_acc = utils.accuracy(outputs, labels)[0].item()
        all_test_acc.append(test_acc)
        arg_idx = outputs.argmax()
        y = outputs[0][arg_idx]
        y.backward()
        saliency = torch.amax(imgs.grad.abs(), dim=1)
        flat_saliency = saliency.flatten()
        topk, indices = torch.topk(flat_saliency, int(medians[img_name]))
        top_k_saliency = torch.zeros(flat_saliency.shape).to(device)
        top_k_saliency = top_k_saliency.scatter_(-1, indices.to(device), topk.to(device))
        top_k_saliency = top_k_saliency.reshape(saliency.shape)
        top_k_saliency = F.interpolate(top_k_saliency.unsqueeze(0), size=(224, 224), mode="bilinear").to(torch.float32)
        hmps = tvF.center_crop(hmps, (224, 224))
        random_hmps = tvF.center_crop(random_hmps, (224, 224))
        top_k_saliency = utils.gaussian_blur(saliency, kernel)
        top_k_saliency = top_k_saliency.detach().cpu().numpy()
        hmps = hmps.detach().cpu().numpy()
        random_hmps = random_hmps.detach().cpu().numpy()
        topk_score, p_value = spearmanr(top_k_saliency.ravel(), hmps.ravel())
        null_score, p_value = spearmanr(top_k_saliency.ravel(), random_hmps.ravel())

        if len(visualize)>0 and i in visualize:
            top_k_img = top_k_saliency.squeeze()
            #top_k_overlay = utils.save_as_overlay(img, top_k_img, os.path.join(args.output_dir,f'topk_blur_{str(i).zfill(3)}.png'))
            hmps_img = hmps.squeeze()
            #hmps_overlay = utils.save_as_overlay(img, hmps_img, os.path.join(args.output_dir,f'hmps_blur_{str(i).zfill(3)}.png'))
            
            f = plt.figure()
            plt.subplot(1, 3, 1)
            top_k_img = (top_k_img - np.min(top_k_img))/np.max(top_k_img)
            plt.imshow(top_k_img)
            plt.axis("off")
            plt.subplot(1, 3, 2)
            hmps_img = (hmps_img - np.min(hmps_img))/np.max(hmps_img)
            plt.imshow(hmps_img)
            plt.axis("off")
            plt.subplot(1, 3, 3)
            plt.imshow(img)
            plt.axis("off")
            f.tight_layout(pad=0)
            f.canvas.draw()
            buf = f.canvas.buffer_rgba()
            ncols, nrows = f.canvas.get_width_height()
            image = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 4)
            image = torch.unsqueeze(torch.Tensor(image), 0)
            image = image[:, int(math.floor(image.shape[1]/4)):int(image.shape[1] - math.floor(image.shape[1]/4)), :, :]
            sample_maps.append(image)

        # category_alignments[cat].append(topk_score/alignments[cat])
        # category_nulls[cat].append(null_score/alignments[cat])
        category_alignments[cat].append(topk_score)
        category_nulls[cat].append(null_score)
        topk_score /= HUMAN_SPEARMAN_CEILING
        null_score /= HUMAN_SPEARMAN_CEILING
        alignment_scores.append(topk_score)
        null_scores.append(null_score)
    
    for cat, cat_align in category_alignments.items():
        category_alignments[cat] = np.mean(cat_align)
        category_nulls[cat] = np.mean(category_nulls[cat])
        category_alignments[cat] = {"alignment": category_alignments[cat], "null": category_nulls[cat]}
    category_alignments['human'] = {"alignment": topk_score, "null": null_score}
    category_json = json.dumps(category_alignments, indent=4)
    with open(os.path.join("data", f'cat_alignment_{str(epoch).zfill(3)}.json'), 'w+') as f:
        f.write(category_json)
    avg_test_acc = sum(all_test_acc)/float(len(all_test_acc))
    wandb.update(heatmaps=sample_maps, head="co3d_eval")
    return avg_test_acc, np.mean(alignment_scores), np.mean(null_scores)

if __name__ == "__main__":
    args = parse_args()
    main(args)