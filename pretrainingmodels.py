from functools import partial
from modeling_finetune import _cfg, get_sinusoid_encoding_table
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from models_3D import PretrainVisionTransformerEncoder, PretrainVisionTransformerDecoder
from models_2D import Masked2DEncoderViT, Masked2DEncoderTIMM
import timm
import torch
import torch.nn as nn
import loralib as lora
from utils import get_cvm_attn_mask

# from utils import _load_weights

def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


__all__ = [
    'e2D_d3D_pretrain_videomae_base_patch16_224',
    'e2D_d3D_pretrain_videomae_small_patch16_224',
    'e3D_d3D_pretrain_videomae_small_patch16_224',
    'e3D_d3D_pretrain_videomae_base_patch16_224',
    'e3D_d3D_pretrain_videomae_large_patch16_224',
    'e3D_d3D_pretrain_videomae_huge_patch16_224',
]


class PretrainVisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 encoder_in_chans=3,
                 encoder_num_classes=0,
                 encoder_embed_dim=768,
                 encoder_depth=12,
                 encoder_num_heads=12,
                 decoder_num_classes=1536, #  decoder_num_classes=768,
                 decoder_embed_dim=512,
                 decoder_depth=8,
                 decoder_num_heads=8,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 use_checkpoint=False,
                 tubelet_size=2,
                 num_classes=0, # avoid the error from create_fn in timm
                 in_chans=0, # avoid the error from create_fn in timm
                 ):

        super().__init__()
        self.encoder = PretrainVisionTransformerEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=encoder_in_chans,
            num_classes=encoder_num_classes,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_checkpoint=use_checkpoint,
            use_learnable_pos_emb=use_learnable_pos_emb)

        self.decoder = PretrainVisionTransformerDecoder(
            patch_size=patch_size,
            num_patches=self.encoder.patch_embed.num_patches,
            num_classes=decoder_num_classes,
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_checkpoint=use_checkpoint)

        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.pos_embed = get_sinusoid_encoding_table(self.encoder.patch_embed.num_patches, decoder_embed_dim)

        trunc_normal_(self.mask_token, std=.02)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def forward(self, x, mask):
        # import pdb; pdb.set_trace()
        _, _, T, _, _ = x.shape
        x_vis = self.encoder(x, mask) # [B, N_vis, C_e]
        x_vis = self.encoder_to_decoder(x_vis) # [B, N_vis, C_d]
        B, N, C = x_vis.shape
        # we don't unshuffle the correct visible token order,
        # but shuffle the pos embedding accorddingly.
        expand_pos_embed = self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C)
        pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, C)
        x_full = torch.cat([x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1) # [B, N, C_d]
        x = self.decoder(x_full, pos_emd_mask.shape[1]) # [B, N_mask, 3 * 16 * 16]

        return x


class Pretrain_2D_3D_VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 encoder_in_chans=3,
                 encoder_num_classes=0,
                 encoder_embed_dim=768,
                 encoder_depth=12,
                 encoder_num_heads=12,
                 decoder_num_classes=768, #  decoder_num_classes=768,
                 decoder_embed_dim=512,
                 decoder_depth=8,
                 decoder_num_heads=8,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 use_checkpoint=False,
                 tubelet_size=1,
                 num_classes=0, # avoid the error from create_fn in timm
                 in_chans=0, # avoid the error from create_fn in timm
                 camera_params_enabled=False,
                 lora_layers = None,
                 lora_attn = "qv",
                 decoder_camera_dropout=0.0,
                 camera_param_dim = 7,
                 return_features = False,
                 use_attn_mask = False,
                 use_register = False,
                 **kwargs
                 ):
        super().__init__()
        self.encoder = Masked2DEncoderViT(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=encoder_in_chans,
            num_classes=encoder_num_classes,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_checkpoint=use_checkpoint,
            use_learnable_pos_emb=use_learnable_pos_emb,
            lora_layers=lora_layers,
            lora_attn=lora_attn)
        # TODO: Change 4 to argument instead of hard code
        self.decoder = PretrainVisionTransformerDecoder(
            patch_size=patch_size,
            num_patches=self.encoder.patch_embed.num_patches*4,
            num_classes=decoder_num_classes,
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_checkpoint=use_checkpoint,
            camera_params_enabled=camera_params_enabled,
            dropout_rate=decoder_camera_dropout,
            camera_param_dim = camera_param_dim)
        if lora_layers is not None:
            lora.mark_only_lora_as_trainable(self.encoder)
        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.pos_embed = get_sinusoid_encoding_table(self.encoder.patch_embed.num_patches*4 + 0, decoder_embed_dim)

        self.return_features = return_features
        trunc_normal_(self.mask_token, std=.02)
        self.mean = [.5,.5,.5]
        self.std = [.5,.5,.5]

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def forward(self, x, mask):
        # import pdb; pdb.set_trace()
        B_orig, _, T, _, _ = x.shape
        # import pdb; pdb.set_trace()
        e_output = self.encoder(x, mask) # [B, N_vis, C_e]
        x_vis = self.encoder_to_decoder(e_output) # [B, N_vis, C_d]
        # import pdb; pdb.set_trace()
        e_output = torch.mean(e_output, dim=1)
        x_cls_token = x_vis[:, :1, :]
        x_vis = x_vis[:, 1:, :]

        _, _, C = x_vis.shape
        x_vis = x_vis.reshape(B_orig, -1, C)
        x_cls_token = x_cls_token.reshape(B_orig, -1, C)
        B, N, C = x_vis.shape
        # we don't unshuffle the correct visible token order,
        # but shuffle the pos embedding accorddingly.
        expand_pos_embed = self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C)
        pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, C)
        x_full = torch.cat([x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1) # [B, N, C_d]

        x = self.decoder(x_full, pos_emd_mask.shape[1]) # [B, N_mask, 3 * 16 * 16]
        if self.decoder.camera_params_enabled:
            x, pc = x
            if self.return_features:
                return x, pc, e_output
            else:
                return x, pc
        # x = x[:, 1:, :] #remove cls
        if self.return_features:
            return x, e_output
        return x

class Pretrain_2D_3D_TIMM_CVM(nn.Module):
    def __init__(self,
                model_name='beit_large_patch16_224.in22k_ft_in22k_in1k',
                patch_size=16,
                encoder_embed_dim=1024,
                decoder_num_classes=768,
                decoder_embed_dim=512,
                decoder_depth=8,
                decoder_num_heads=8,
                mlp_ratio=4.,
                qkv_bias=False,
                qk_scale=None,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.,
                norm_layer=nn.LayerNorm,
                init_values=0.,
                use_learnable_pos_emb=False,
                use_checkpoint=False,
                tubelet_size=1,
                num_classes=0, # avoid the error from create_fn in timm
                in_chans=0, # avoid the error from create_fn in timm
                pretrained=True,
                camera_params_enabled=False,
                lora_layers = None,
                lora_attn = "qv",
                decoder_camera_dropout=0.0,
                camera_param_dim = 7,
                return_features = False,
                num_frames = 4,
                **kwargs
                ):
        super().__init__()
        self.encoder = Masked2DEncoderTIMM(model_name=model_name, pretrained=pretrained)
        self.model_name = model_name
        self.num_frames = num_frames

        self.decoder = PretrainVisionTransformerDecoder(
            patch_size=patch_size,
            num_patches=self.encoder.patch_embed.num_patches*num_frames,
            num_classes=decoder_num_classes,
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_checkpoint=use_checkpoint,
            camera_params_enabled=camera_params_enabled,
            dropout_rate=decoder_camera_dropout,
            camera_param_dim = camera_param_dim)

        if lora_layers is not None:
            lora.mark_only_lora_as_trainable(self.encoder)
        config = timm.data.resolve_model_data_config(self.encoder.model)
        self.mean = config['mean']
        self.std = config['std']
        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)

        #self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        #self.pos_embed = get_sinusoid_encoding_table(self.encoder.patch_embed.num_patches*num_frames + 0, decoder_embed_dim)
        self.pos_embed = get_sinusoid_encoding_table(self.encoder.patch_embed.num_patches*(num_frames-1) + 0, decoder_embed_dim)
        self.return_features = return_features

        #trunc_normal_(self.mask_token, std=.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def forward(self, x, mask):
        B_orig, C, T, H, W = x.shape
        x = x[:, :, :self.num_frames-1, :, :]
        x = x.movedim(1, 2).reshape(B_orig*(T-1), C, H, W)
        e_output = self.encoder(x, mask, f2d=True)
        x_vis = self.encoder_to_decoder(e_output) # [B, N_vis, C_d]
        e_output = torch.mean(e_output, dim=1)
        x_cls_token = x_vis[:, :1, :]
        x_vis = x_vis[:, 1:, :]

        _, _, C = x_vis.shape
        x_vis = x_vis.reshape(B_orig, -1, C) #(B, (T-1)*N_e, C)
        #x_cls_token = x_cls_token.reshape(B_orig, -1, C)
        B, N, C = x_vis.shape
        # we don't unshuffle the correct visible token order,
        # but shuffle the pos embedding accorddingly.
        expand_pos_embed = self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        #pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C)
        #pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, C)
        #x_full = torch.cat([x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1) # [B, N, C_d]
        x_vis = x_vis + expand_pos_embed
        x_vis = x_vis.reshape(B_orig*(T-1), -1, C)
        x_full = torch.cat((x_cls_token, x_vis), dim=1).reshape(B_orig, -1, C)
        attn_mask = get_cvm_attn_mask(self.encoder.patch_embed.num_patches*(self.num_frames-1)+(self.num_frames-1), self.num_frames-1).to(x.device)
        x = self.decoder(x_full, 0, attn_mask) #(B, N+T-1, 768*16)
        B, N, C = x.shape
        x = x.reshape(B_orig*(T-1), -1, C)[:, 1:, :].reshape(B_orig, -1, C)
        if self.decoder.camera_params_enabled:
            x, pc = x
            if self.return_features:
                return x, pc, e_output
            else:
                return x, pc
        # x = x[:, 1:, :] #remove cls
        if self.return_features:
            return x, e_output
        return x

class Pretrain_2D_3D_TIMM(nn.Module):
    def __init__(self,
                model_name='beit_large_patch16_224.in22k_ft_in22k_in1k',
                patch_size=16,
                encoder_embed_dim=1024,
                decoder_num_classes=768,
                decoder_embed_dim=512,
                decoder_depth=8,
                decoder_num_heads=8,
                mlp_ratio=4.,
                qkv_bias=False,
                qk_scale=None,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.,
                norm_layer=nn.LayerNorm,
                init_values=0.,
                use_learnable_pos_emb=False,
                use_checkpoint=False,
                tubelet_size=1,
                num_classes=0, # avoid the error from create_fn in timm
                in_chans=0, # avoid the error from create_fn in timm
                pretrained=True,
                camera_params_enabled=False,
                lora_layers = None,
                lora_attn = "qv",
                decoder_camera_dropout=0.0,
                camera_param_dim = 7,
                return_features = False,
                **kwargs
                ):
        super().__init__()
        self.encoder = Masked2DEncoderTIMM(model_name=model_name, pretrained=pretrained)

        # TODO: Change 4 to argument instead of hard code
        self.decoder = PretrainVisionTransformerDecoder(
            patch_size=patch_size,
            num_patches=self.encoder.patch_embed.num_patches*4,
            num_classes=decoder_num_classes,
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_checkpoint=use_checkpoint,
            camera_params_enabled=camera_params_enabled,
            dropout_rate=decoder_camera_dropout,
            camera_param_dim = camera_param_dim)

        if lora_layers is not None:
            lora.mark_only_lora_as_trainable(self.encoder)
        config = timm.data.resolve_model_data_config(self.encoder.model)
        self.mean = config['mean']
        self.std = config['std']
        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.pos_embed = get_sinusoid_encoding_table(self.encoder.patch_embed.num_patches*4 + 0, decoder_embed_dim)
        self.return_features = return_features

        trunc_normal_(self.mask_token, std=.02)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def forward(self, x, mask):
        # import pdb; pdb.set_trace()
        B_orig, _, T, _, _ = x.shape
        # import pdb; pdb.set_trace()
        e_output = self.encoder(x, mask) # [B, N_vis, C_e]
        x_vis = self.encoder_to_decoder(e_output) # [B, N_vis, C_d]
        # import pdb; pdb.set_trace()
        e_output = torch.mean(e_output, dim=1)
        x_cls_token = x_vis[:, :1, :]
        x_vis = x_vis[:, 1:, :]

        _, _, C = x_vis.shape
        x_vis = x_vis.reshape(B_orig, -1, C)
        #x_cls_token = x_cls_token.reshape(B_orig, -1, C)
        B, N, C = x_vis.shape
        # we don't unshuffle the correct visible token order,
        # but shuffle the pos embedding accorddingly.
        expand_pos_embed = self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C)
        pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, C)
        x_full = torch.cat([x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1) # [B, N, C_d]

        x = self.decoder(x_full, pos_emd_mask.shape[1]) # [B, N_mask, 3 * 16 * 16]
        if self.decoder.camera_params_enabled:
            x, pc = x
            if self.return_features:
                return x, pc, e_output
            else:
                return x, pc
        # x = x[:, 1:, :] #remove cls
        if self.return_features:
            return x, e_output
        return x

class Pretrain_2D_2D_TIMM(nn.Module):
    def __init__(self,
                model_name='beit_large_patch16_224.in22k_ft_in22k_in1k',
                patch_size=16,
                encoder_embed_dim=1024,
                decoder_num_classes=768,
                decoder_embed_dim=512,
                decoder_depth=8,
                decoder_num_heads=8,
                mlp_ratio=4.,
                qkv_bias=False,
                qk_scale=None,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.,
                norm_layer=nn.LayerNorm,
                init_values=0.,
                use_learnable_pos_emb=False,
                use_checkpoint=False,
                tubelet_size=1,
                num_classes=0, # avoid the error from create_fn in timm
                in_chans=0, # avoid the error from create_fn in timm
                pretrained=True,
                camera_params_enabled=False,
                lora_layers = None,
                lora_attn = "qv",
                decoder_camera_dropout=0.0,
                camera_param_dim = 7,
                return_features = False,
                **kwargs
                ):
        super().__init__()
        self.encoder = Masked2DEncoderTIMM(model_name=model_name, pretrained=pretrained)

        self.decoder = PretrainVisionTransformerDecoder(
            patch_size=patch_size,
            num_patches=self.encoder.patch_embed.num_patches*4,
            num_classes=decoder_num_classes,
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_checkpoint=use_checkpoint,
            camera_params_enabled=camera_params_enabled,
            dropout_rate=decoder_camera_dropout,
            camera_param_dim = camera_param_dim)

        config = timm.data.resolve_model_data_config(self.encoder.model)
        self.mean = config['mean']
        self.std = config['std']
        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.pos_embed = get_sinusoid_encoding_table(self.encoder.patch_embed.num_patches*1 + 1, decoder_embed_dim)
        self.return_features = return_features

        trunc_normal_(self.mask_token, std=.02)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def forward(self, x, mask):
        # import pdb; pdb.set_trace()
        B_orig, C_orig, T, H, W = x.shape
        # import pdb; pdb.set_trace()
        e_output = self.encoder(x, mask) # [B, N_vis, C_e]
        x_vis = self.encoder_to_decoder(e_output) # [B, N_vis, C_d]
        # import pdb; pdb.set_trace()
        e_output = torch.mean(e_output, dim=1)
        x_cls_token = x_vis[:, :1, :]
        x_vis = x_vis[:, 1:, :]
        B, N, C = x_vis.shape
        mask = mask.view(B, -1)
        expand_pos_embed = self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        x_cls_token = x_cls_token + expand_pos_embed[:, :1, :]
        expand_pos_embed = expand_pos_embed[:, 1:, :]
        pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C)
        pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, C)
        x_full = torch.cat([x_cls_token, x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1) # [B, N, C_d]

        x = self.decoder(x_full, pos_emd_mask.shape[1]) # [B, N_mask, 3 * 16 * 16]
        _, _, C = x.shape
        x = x.reshape(B_orig, -1, C)
        if self.decoder.camera_params_enabled:
            x, pc = x
            if self.return_features:
                return x, pc, e_output
            else:
                return x, pc
        # x = x[:, 1:, :] #remove cls

        if self.return_features:
            return x, e_output
        return x


class Pretrain_2D_2D_VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 encoder_in_chans=3,
                 encoder_num_classes=0,
                 encoder_embed_dim=768,
                 encoder_depth=12,
                 encoder_num_heads=12,
                 decoder_num_classes=768, #  decoder_num_classes=768,
                 decoder_embed_dim=512,
                 decoder_depth=8,
                 decoder_num_heads=8,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 use_checkpoint=False,
                 tubelet_size=1,
                 num_classes=0, # avoid the error from create_fn in timm
                 in_chans=0, # avoid the error from create_fn in timm
                 **kwargs
                 ):
        super().__init__()
        self.encoder = Masked2DEncoderViT(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=encoder_in_chans,
            num_classes=encoder_num_classes,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_checkpoint=use_checkpoint,
            use_learnable_pos_emb=use_learnable_pos_emb)

        self.decoder = PretrainVisionTransformerDecoder(
            patch_size=patch_size,
            num_patches=self.encoder.patch_embed.num_patches,
            num_classes=decoder_num_classes,
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_checkpoint=use_checkpoint)

        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.pos_embed = get_sinusoid_encoding_table(self.encoder.patch_embed.num_patches*1 + 1, decoder_embed_dim)

        trunc_normal_(self.mask_token, std=.02)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def forward(self, x, mask):
        # import pdb; pdb.set_trace()
        B_orig, C_orig, T, H, W = x.shape
        x_vis = self.encoder(x, mask) # [B, N_vis, C_e]
        x_vis = self.encoder_to_decoder(x_vis) # [B, N_vis, C_d]
        # import pdb; pdb.set_trace()

        x_cls_token = x_vis[:, :1, :]
        x_vis = x_vis[:, 1:, :]
        
        B, N, C = x_vis.shape
        mask = mask.view(B, -1)
        # we don't unshuffle the correct visible token order,
        # but shuffle the pos embedding accorddingly.
        expand_pos_embed = self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        x_cls_token = x_cls_token + expand_pos_embed[:, :1, :]
        expand_pos_embed = expand_pos_embed[:, 1:, :]
        pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C)
        pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, C)
        x_full = torch.cat([x_cls_token, x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1) # [B, N, C_d]
        x = self.decoder(x_full, pos_emd_mask.shape[1]) # [B, N_mask, 3 * 16 * 16]
        # x = x[:, 1:, :] #remove cls
        _, _, C = x.shape
        x = x.reshape(B_orig, -1, C) #reshape back to tensor with time dim to viz etc
        return x


@register_model
def e2D_d3D_pretrain_videomae_beit_large(pretrained=True, **kwargs):
    print(kwargs)
    model = Pretrain_2D_3D_TIMM(
        model_name='beit_large_patch16_224.in22k_ft_in22k_in1k',
        patch_size=16,
        encoder_embed_dim=1024,
        decoder_embed_dim=512,
        decoder_num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        pretrained=pretrained,
        **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def e2D_d3D_pretrain_videomae_small_patch16_224_cvm(pretrained=True, **kwargs):
    print(kwargs)
    model = Pretrain_2D_3D_TIMM_CVM(
        model_name = 'vit_small_patch16_224.augreg_in21k_ft_in1k',
        patch_size=16,
        encoder_embed_dim=384,
        # decoder_num_classes=768,
        decoder_embed_dim=192,
        decoder_num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        pretrained=pretrained,
        **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def e2D_d3D_pretrain_videomae_small_patch16_224_timm(pretrained=True, **kwargs):
    print(kwargs)
    model = Pretrain_2D_3D_TIMM(
        model_name = 'vit_small_patch16_224.augreg_in21k_ft_in1k',
        patch_size=16,
        encoder_embed_dim=384,
        # decoder_num_classes=768,
        decoder_embed_dim=192,
        decoder_num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        pretrained=pretrained,
        **kwargs)
    model.default_cfg = _cfg()
    return model

    
@register_model
def e2D_d3D_pretrain_videomae_base_patch16_224(pretrained=False, **kwargs):
    print(kwargs)
    model = Pretrain_2D_3D_VisionTransformer(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_num_classes=0,
        decoder_num_classes=768,
        decoder_embed_dim=384,
        decoder_num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        print("Loading weights in pretrainingmodels.py e2D_d3D_pretrain_videomae_base_patch16_224")
        checkpoint = torch.load(kwargs['ckpt_path'], map_location="cpu")
        model.load_state_dict(checkpoint)
    return model


@register_model
def e2D_d3D_pretrain_videomae_small_patch16_224(pretrained=False, **kwargs):
    print(kwargs)
    model = Pretrain_2D_3D_VisionTransformer(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=384,
        encoder_depth=12,
        encoder_num_heads=6,
        encoder_num_classes=0,
        # decoder_num_classes=768,
        decoder_embed_dim=192,
        decoder_num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        print("Loading weights in pretrainingmodels.py e2D_d3D_pretrain_videomae_small_patch16_224")
        checkpoint = torch.load(kwargs['ckpt_path'], map_location="cpu")
        print(model.encoder)
        #checkpoint = {k: v for k, v in checkpoint.items() if k in model.encoder.state_dict().keys()}
        msg = model.encoder.load_state_dict(checkpoint, strict=False)
        print("Weight Loading Status:", msg)
    return model


@register_model
def e2D_d2D_pretrain_videomae_small_patch16_224(pretrained=False, **kwargs):
    print(kwargs)
    model = Pretrain_2D_2D_VisionTransformer(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=384,
        encoder_depth=12,
        encoder_num_heads=6,
        encoder_num_classes=0,
        decoder_num_classes=768,
        decoder_embed_dim=192,
        decoder_num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        print("Loading weights in pretrainingmodels.py e2D_d2D_pretrain_videomae_small_patch16_224")
        checkpoint = torch.load(kwargs['ckpt_path'], map_location="cpu")
        model.load_state_dict(checkpoint)
    return model

@register_model
def e2D_d2D_pretrain_videomae_small_patch16_224_timm(pretrained=True, **kwargs):
    print(kwargs)
    model = Pretrain_2D_2D_TIMM(
        model_name = 'vit_small_patch16_224.augreg_in21k_ft_in1k',
        patch_size=16,
        encoder_embed_dim=384,
        # decoder_num_classes=768,
        decoder_embed_dim=192,
        decoder_num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        pretrained=pretrained,
        **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def e2D_d2D_pretrain_videomae_base_patch16_224(pretrained=False, **kwargs):
    model = Pretrain_2D_2D_VisionTransformer(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_num_classes=0,
        decoder_num_classes=768,
        decoder_embed_dim=384,
        decoder_num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        print("Loading weights in pretrainingmodels.py e2D_d2D_pretrain_videomae_base_patch16_224")
        checkpoint = torch.load(kwargs['ckpt_path'], map_location="cpu")
        model.load_state_dict(checkpoint)
    return model


@register_model
def e3D_d3D_pretrain_videomae_small_patch16_224(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=384,
        encoder_depth=12,
        encoder_num_heads=6,
        encoder_num_classes=0,
        decoder_num_classes=1536,
        decoder_embed_dim=192,
        decoder_num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs['ckpt_path'], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def e3D_d3D_pretrain_videomae_base_patch16_224(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_num_classes=0,
        decoder_num_classes=1536,
        decoder_embed_dim=384,
        decoder_num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs['ckpt_path'], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def e3D_d3D_pretrain_videomae_large_patch16_224(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_num_classes=0,
        decoder_num_classes=1536,
        decoder_embed_dim=512,
        decoder_num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs['ckpt_path'], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def e3D_d3D_pretrain_videomae_huge_patch16_224(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_num_classes=0,
        decoder_num_classes=1536,
        decoder_embed_dim=640,
        decoder_num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs['ckpt_path'], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model
