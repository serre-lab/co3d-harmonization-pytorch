from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

from utils import get_2d_sincos_pos_embed
import timm

class Masked2DEncoderTIMM(nn.Module):
    def __init__(self, model_name='beit_large_patch16_224.in22k_ft_in22k_in1k', pretrained=True, **kwargs):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            )
        self.pretrained=pretrained
        self.model_name = model_name
        self.patch_embed = self.model.patch_embed
        # self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, self.model.num_features), requires_grad=False)  # fixed sin-cos embedding
        self.embed_dim = self.model.num_features
        # self.norm = self.model.norm
        # self.cls_token = self.model.cls_token
        #self.model.fc_norm = None
        #self.initialize_weights()

    def forward_encoder(self, x, mask):

        if len(x.shape) == 5:
            B, C, T, H, W = x.shape
            x = x.movedim(1, 2).reshape(B*T, C, H, W) # 2d encoder process frames separately   
        x = self.model.forward_features(x)
        B, _, C = x.shape
        selector = ~mask.view(B, -1)
        cls_token = torch.unsqueeze(x[:, 0, :], 1)
        x = x[:, 1:, :]
        cls_selector = torch.unsqueeze(torch.max(selector, dim=1)[0], 1)
        if (~selector[0]).sum() == 0:
            B = int(B*0.75)
        cls_token = cls_token[cls_selector].reshape(B, -1, C)

        x = x[selector].reshape(B, -1, C)
        x = torch.cat((cls_token, x), dim=1)
        return x

    def forward_2D(self, x):
        x = self.model.forward_features(x)
        #x = global_pool_nlc(x, self.model.pool_type)
        #x = self.model(x)
        return x

    def forward(self, imgs, mask=None, f2d=False):
        if f2d:
            latent = self.forward_2D(imgs)
        else:
            latent = self.forward_encoder(imgs, mask)
        return latent

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        if not self.pretrained:
            # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
            w = self.patch_embed.proj.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

            # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
            torch.nn.init.normal_(self.cls_token, std=.02)
            # torch.nn.init.normal_(self.mask_token, std=.02)

            # initialize nn.Linear and nn.LayerNorm
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class Masked2DEncoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, lora_layers=None, 
                 lora_attn="qv", **kwargs):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.embed_dim = embed_dim
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])

        if lora_layers is not None:
            if lora_layers == 'all':
                lora_layers = list(range(depth))
            elif lora_layers == 'last':
                lora_layers = [-1]
            else:
                lora_layers = [int(item) for item in lora_layers.split(',')]
            if lora_attn != "mlp":
                enable_lora = ["q" in lora_attn, "k" in lora_attn, "v" in lora_attn]
                for i in lora_layers:
                    self.blocks[i].attn.qkv = lora.MergedLinear(embed_dim, 3*embed_dim, r=8, enable_lora=enable_lora,
                                    bias=True)
            else:
                for i in lora_layers:
                    self.blocks[i].mlp.fc1 = lora.Linear(embed_dim, int(embed_dim*mlp_ratio), bias=True, r=8)
                    self.blocks[i].mlp.fc2 = lora.Linear(int(embed_dim*mlp_ratio), embed_dim, bias=True, r=8)
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))


        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        # torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, x, mask):
        # embed patches
        if len(x.shape) == 5:  #multiview case -alekh
            B, C, T, H, W = x.shape
            x = x.movedim(1, 2).reshape(B*T, C, H, W) # 2d encoder process frames separately
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        B, _, C = x.shape #B = B*T | C --> enc emd dim
        selector = ~mask.view(B, -1)
        if (~selector[0]).sum() == 0:  #causal mode
            B = int(B*0.75)
        x = x[selector].reshape(B, -1, C)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x #, mask, ids_restore

    def forward_2D(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def forward(self, imgs, mask=None, f2d=False):
        if f2d:
            latent = self.forward_2D(imgs)
        else:
            latent = self.forward_encoder(imgs, mask)
        return latent

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