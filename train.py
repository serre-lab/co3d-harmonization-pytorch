import argparse
import torch
import math
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torch.utils.data import DataLoader
from dataset import ClickMe, EmbeddingDataset, build_co3d_eval_loader
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
from typing import Iterable
import pretrainingmodels

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
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers in the system")
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
    train_dataset = ClickMe(image_folder="data/CO3D_ClickMe2/", label_to_category_map=label_to_category_map, is_training=True)
    print("Created training dataset, size:", len(train_dataset))
    
    val_dataset = ClickMe(image_folder="data/CO3D_ClickMe2/", label_to_category_map=label_to_category_map, is_training=False)
    print("Created validation dataset, size:", len(val_dataset))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    co3d_train_dataloader, co3d_val_dataloader, _ = build_co3d_eval_loader(args, None, True, label_to_index_map=label_to_category_map)
    co3d_test_dataloader = build_co3d_eval_loader(args, None, False, label_to_index_map=label_to_category_map)

    # Load the ViT small model
    model = timm.create_model('vit_small_patch16_224', pretrained=True)
    p_model = timm.create_model(
        "e2D_d3D_pretrain_videomae_small_patch16_224",
        pretrained=True,
        drop_path_rate=0.0,
        drop_block_rate=None,
        decoder_depth=4,
        decoder_num_classes=768,
        use_checkpoint=True,
        camera_params_enabled=False,
        ckpt_path='/cifs/data/tserre_lrs/projects/projects/prj_video_imagenet/TempAkash/vit_small16_timm_weights.bin',
        lora_layers=None,
        lora_attn="qv",
        decoder_camera_dropout=0.0,
        camera_param_dim=7,
        return_features=True,
        num_frames = 4
    )
    model.heads = nn.Linear(model.embed_dim, N_CO3D_CLASSES)

    linear_model = LinearModel(p_model.encoder.embed_dim, 51)
    linear_optimizer = torch.optim.Adam(linear_model.parameters(), lr=0.001)
    
    model = model.to(device)

    # Define optimizer and custom loss
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    

    gaussian_kernel = utils.gaussian_kernel(size=BRUSH_SIZE, sigma=BRUSH_SIZE_SIGMA).to(device)

    print(f"Image 1 of Training Set: {train_dataset[0][0].shape}")
    print(f"Image 1 of Validation Set: {val_dataset[0][0].shape}")

    image_with_heatmap_count = 0
    for d in train_dataset.data:
        if len(d['heatmap']) > 0:
            image_with_heatmap_count += 1
    print("Images with heatmaps in training set:", image_with_heatmap_count)

    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0.0
        total_cce_loss = 0.0
        total_harmonization_loss = 0.0
        train_acc = 0.0
        train_correct = 0
        total_samples = 0

        batch_count = 0

        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{args.num_epochs}"):
            # print(f"Batch: {batch}")
            images, heatmaps, targets, has_heatmap_batch = batch[0], batch[1], batch[2], batch[3]
            images, heatmaps, targets = images.to(device), heatmaps.to(device), targets.to(device)

            # image_count = 0
            # heatmap_count = 0
            # for i, h, t, hh in zip(images, heatmaps, targets, has_heatmap_batch):
            #     if hh:
            #         heatmap_count += 1
            #         print(f"Image {image_count} has heatmap")
            #     image_count += 1
            # # print(f"Batch {batch_count} has {image_count} images, {heatmap_count} heatmaps")
            # batch_count += 1

            images.requires_grad = True

            optimizer.zero_grad()

            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                y_pred = model(images) 
                most_probable_class = torch.argmax(y_pred, dim=-1)  # Get index of the most probable class for each image
                # Scores of the most probable class for each image
                most_probable_scores = y_pred[torch.arange(y_pred.size(0)), most_probable_class]  # Shape: (batch_size,)
                saliency_map = torch.autograd.grad(outputs=most_probable_scores, inputs=images,
                                                grad_outputs=torch.ones_like(most_probable_scores),
                                                create_graph=True)[0]
                saliency_map = saliency_map.mean(dim=1)  # Averaging across the channels to get shape: (batch_size, height, width)
                        
                heatmap = utils.gaussian_blur(heatmaps.unsqueeze(1), gaussian_kernel).unsqueeze(1)
                saliency_map = utils.gaussian_blur(saliency_map.unsqueeze(1), gaussian_kernel).unsqueeze(1)

                # harmonization_loss, _ = harmonizer_loss(y_pred, targets, heatmap, saliency_map)


                heatmap_count = 0 + 1e-6
                # total_cce_loss = 0.0
                # total_harmonization_loss = 0.0
                # for image, heatmap, target, predicted_label, has_heatmap, most_probable_score in zip(images, heatmaps, targets, y_pred, has_heatmap_batch, most_probable_scores):            
                #     if has_heatmap:
                #         heatmap_count += 1
                #         saliency_map = torch.autograd.grad(outputs=most_probable_score, inputs=image,
                #                                 grad_outputs=torch.ones_like(most_probable_score),
                #                                 create_graph=True)[0]
                #         saliency_map = saliency_map.mean(dim=1)  # Averaging across the channels to get shape: (batch_size, height, width)
                        
                #         heatmap = utils.gaussian_blur(heatmaps.unsqueeze(1), gaussian_kernel).unsqueeze(1)
                #         saliency_map = utils.gaussian_blur(saliency_map.unsqueeze(1), gaussian_kernel).unsqueeze(1)

                #         harmonization_loss, _ = harmonizer_loss(predicted_label, target, heatmap, saliency_map)
                # total_harmonization_loss += harmonization_loss
                total_cce_loss = nn.CrossEntropyLoss()(y_pred, targets)

                # print(f"Calculated number of heatmaps: {heatmap_count} for {len(images)} images")  

                total_loss = total_cce_loss/len(images)
                # if heatmap_count > 0:
                # total_loss += total_harmonization_loss/heatmap_count

                total_loss.backward()

                optimizer.step()

            train_loss += total_loss.item()
            total_cce_loss += total_cce_loss.item()
            # total_harmonization_loss += total_harmonization_loss.item()
            train_correct += (torch.argmax(y_pred, dim=1) == targets).sum().item()
            total_samples += targets.size(0)
    
        train_acc = train_correct / total_samples
         
        print(f"Epoch [{epoch+1}/{args.num_epochs}], Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}")
        wandb.log({"train_loss": train_loss/len(train_loader), "train_acc": train_acc, "cce_loss": total_cce_loss, "harmonization_loss": total_harmonization_loss})
        # eval_alignment(model, val_loader, device, epoch, None, list(range(10)), args)
        eval_co3d(p_model, co3d_train_dataloader, co3d_val_dataloader, co3d_test_dataloader, device, epoch,
                30, 256, 5e-4, None, start_steps=0, num_workers=8, args=args)
            #   f"Val Loss: {val_loss/len(val_loader):.4f}")

    print(f"Training completed. Model saved to {args.output_model}")

def extract_features(model: torch.nn.Module, data_loader: Iterable, device: torch.device):
    model.eval()
    model.to(device)
    features = []
    labels_list = []
    for i, data in enumerate(tqdm(data_loader)):
        with torch.no_grad():
            images, labels = data
            images = images.to(device)
            preds = model(images, f2d=True)
            if len(preds.shape) > 2:
                preds = torch.mean(preds, dim=1)
            features.append(preds.cpu())
            labels_list.append(labels.cpu())

    features = torch.cat(features)
    labels = torch.cat(labels_list).squeeze()
    return features, labels

class LinearModel(torch.nn.Module):
    def __init__(self, input_dim, num_classes, dropout_rate=0.7):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.linear = torch.nn.Linear(input_dim, num_classes)
        self.batchnorm = torch.nn.BatchNorm1d(input_dim, affine=False, eps=1e-6)
    def forward(self, x):
        x = self.batchnorm(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x
    
class FullModel(nn.Module):
    def __init__(self, encoder, head):
        super(FullModel, self).__init__()
        self.head = head
        self.encoder = encoder

    def forward(self, x):
        x = self.encoder(x, f2d=True)
        if len(x.shape) > 2:
            x = x.mean(dim=1)
        return self.head(x)
    
def eval_co3d(model: torch.nn.Module, train_data_loader: Iterable, val_data_loader: Iterable, test_data_loader: Iterable, device: torch.device, epoch: int, num_epochs: int, 
                batch_size: int, learning_rate=5e-4, log_writer=None, start_steps=None, num_workers=16, args=None):
    train_features, train_labels = extract_features(model.encoder, train_data_loader, device)
    val_features, val_labels = extract_features(model.encoder, val_data_loader, device)

    metric_logger = utils.MetricLogger(delimiter="   ")
    header = f'Co3D EVAL'
    print_freq = 10
    
    train_dataset = EmbeddingDataset(train_features, train_labels)
    val_dataset = EmbeddingDataset(val_features, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    linear_model = LinearModel(input_dim=train_features.shape[-1], num_classes = len(set(train_labels.numpy())), dropout_rate=0.5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(linear_model.parameters(), lr=learning_rate, weight_decay=1e-4)
    best_acc = 0
    for e in metric_logger.log_every(range(num_epochs), print_freq, header):
        metric_logger.update(epoch=e)
        linear_model.train()
        train_loss = 0
        for batch in train_loader:
            embeddings, labels = batch
            embeddings = embeddings.to(device)
            labels = labels.to(device).type(torch.long)
            preds = linear_model(embeddings)
            loss = criterion(preds, labels)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        avg_train_loss = train_loss / len(train_loader)

        linear_model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                embeddings, labels = batch
                embeddings = embeddings.to(device)
                labels = labels.to(device)
                preds = linear_model(embeddings)
                loss = criterion(preds, labels)
                val_loss += loss.item()
                _, predicted = torch.max(preds.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        avg_val_loss = val_loss / len(val_loader)
        acc = 100 * correct/total
        best_acc = max(acc, best_acc)
 
        metric_logger.update(train_loss=avg_train_loss)
        metric_logger.update(val_loss=avg_val_loss)
        metric_logger.update(acc=acc)


    #TODO Use model with best val acc for alignment and test
    full_model = FullModel(model.encoder, linear_model)
    avg_test_acc, avg_alignment, null_alignment = eval_alignment(full_model, test_data_loader, device, epoch, log_writer, list(range(10)), args)
    if log_writer is not None:
        log_writer.update(test_acc=avg_test_acc, head='co3d_eval')
        log_writer.update(alignment=avg_alignment, head='co3d_eval')
        log_writer.update(null_align=null_alignment, head='co3d_eval')
        log_writer.update(val_acc=best_acc, head='co3d_eval')
        log_writer.update(epoch=epoch, head='co3d_eval')
        log_writer.update(commit=True, grad_norm=0, head="co3d_eval")


    return

def eval_alignment(full_model:torch.nn.Module, test_data_loader: Iterable, 
                device: torch.device, epoch: int, log_writer, visualize, args):
    #TODO Save image separately for each epoch. log into wandb
    sample_imgs = []
    sample_maps = []
    sample_clickmaps = []
    with open(args.medians_json, 'r') as f:
        medians = json.load(f)
    with open(args.alignments_json, 'r') as f:
        alignments = json.load(f)
    criterion = nn.CrossEntropyLoss()
    #TODO Use model with best val acc for alignment and test
    HUMAN_SPEARMAN_CEILING = alignments['human']
    alignment_scores = []
    null_scores = []
    all_test_acc = []
    category_alignments = {}
    category_nulls = {}
    kernel = utils.gaussian_kernel(size=11, sigma=math.sqrt(11)).to(device)
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
            img = img*args.std + args.mean
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
    with open(os.path.join(args.output_dir, f'cat_alignment_{str(epoch).zfill(3)}.json'), 'w') as f:
        f.write(category_json)
    avg_test_acc = sum(all_test_acc)/float(len(all_test_acc))
    if log_writer is not None:
        log_writer.update(heatmaps=sample_maps, head="co3d_eval")
    print(f"Alignment Score: {np.mean(alignment_scores)}, Test Accuracy: {np.mean(all_test_acc)}")
    wandb.log({"alignment_score": np.mean(alignment_scores), "test_acc": np.mean(all_test_acc), "epoch": epoch})
    return avg_test_acc, np.mean(alignment_scores), np.mean(null_scores)


if __name__ == "__main__":
    args = parse_args()
    main(args)

            # # 2. Second Approach (Tony's):
            # correct_class_scores = y_pred.gather(1, targets.view(-1, 1)).squeeze()
            # device = images.device
            # ones_tensor = torch.ones(correct_class_scores.shape).to(device) # scores is a tensor here, need to supply initial gradients of same tensor shape as scores.
            # # Saliency map
            # grads = torch.autograd.grad(outputs=correct_class_scores, inputs=images, grad_outputs=ones_tensor, retain_graph=True, create_graph=True, only_inputs=True)[0]
            # saliency_maps = torch.mean(grads, dim=1, keepdim=True)

            # # 3. Third Approach (from scratch):
            # arg_idx = torch.argmax(y_pred, dim=1)  # Get the index of the most probable class for each image in the batch
            # y = y_pred[torch.arange(y_pred.size(0)), arg_idx]  # Get the predicted score of the most probable class
            # # Compute the gradient of the sum of the most probable class scores w.r.t. the input images
            # y.sum().backward(retain_graph=True)
            # # # The gradients w.r.t. the input images
            # saliency_maps = images.grad  # Shape: (batch_size, channels, height, width)
            # # # Mean of the gradients across channels to get a 2D saliency map
            # saliency_maps = saliency_maps.mean(dim=1)  # Shape: (batch_size, height, width)

            #     heatmaps = utils.gaussian_blur(heatmaps.unsqueeze(1), gaussian_kernel).unsqueeze(1)
            #     saliency_maps = utils.gaussian_blur(saliency_maps.unsqueeze(1), gaussian_kernel).unsqueeze(1)

            #  # Index 0 -> harmonization loss, Index 1 -> cross entropy loss
            # harmonization_loss, cce_loss = harmonizer_loss(y_pred, targets, heatmaps, saliency_maps)
            # total_loss = harmonization_loss + cce_loss
            # total_loss.backward()
