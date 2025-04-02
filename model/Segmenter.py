import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
import torch
from einops import rearrange
import torch.nn.functional as F
from .pos_embed import get_2d_sincos_pos_embed, get_2d_sincos_pos_embed_from_grid, get_1d_sincos_pos_embed_from_grid

def init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout, out_dim=None):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        if out_dim is None:
            out_dim = dim
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.attn = None

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.heads, C // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout, drop_path):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, mask=None, return_attention=False):
        y, attn = self.attn(self.norm1(x), mask)
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class DecoderLinear(nn.Module):
    def __init__(self, n_cls, patch_size, d_encoder):
        super().__init__()

        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_cls = n_cls

        self.head = nn.Linear(self.d_encoder, n_cls)
        self.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward(self, x, im_size):
        H, W = im_size
        GS = H // self.patch_size
        x = self.head(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=GS)

        return x


class MaskTransformer(nn.Module):
    def __init__(
        self,
        n_cls,
        patch_size,
        d_encoder,
        n_layers,
        n_heads,
        d_model,
        d_ff,
        drop_path_rate,
        dropout,
    ):
        super().__init__()
        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.n_cls = n_cls
        self.d_model = d_model
        self.d_ff = d_ff
        self.scale = d_model ** -0.5

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
        )

        self.cls_emb = nn.Parameter(torch.randn(1, n_cls, d_model))
        self.proj_dec = nn.Linear(d_encoder, d_model)

        self.proj_patch = nn.Parameter(self.scale * torch.randn(d_model, d_model))
        self.proj_classes = nn.Parameter(self.scale * torch.randn(d_model, d_model))

        self.decoder_norm = nn.LayerNorm(d_model)
        self.mask_norm = nn.LayerNorm(n_cls)

        self.apply(init_weights)
        trunc_normal_(self.cls_emb, std=0.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_emb"}

    def forward(self, x, im_size):
        H, W = im_size
        GS = H // self.patch_size

        x = self.proj_dec(x)
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        x = torch.cat((x, cls_emb), 1)
        for blk in self.blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        patches, cls_seg_feat = x[:, : -self.n_cls], x[:, -self.n_cls :]
        patches = patches @ self.proj_patch
        cls_seg_feat = cls_seg_feat @ self.proj_classes

        patches = patches / patches.norm(dim=-1, keepdim=True)
        cls_seg_feat = cls_seg_feat / cls_seg_feat.norm(dim=-1, keepdim=True)

        masks = patches @ cls_seg_feat.transpose(1, 2)
        masks = self.mask_norm(masks)
        masks = rearrange(masks, "b (h w) n -> b n h w", h=int(GS))

        return masks

    def get_attention_map(self, x, layer_id):
        if layer_id >= self.n_layers or layer_id < 0:
            raise ValueError(
                f"Provided layer_id: {layer_id} is not valid. 0 <= {layer_id} < {self.n_layers}."
            )
        x = self.proj_dec(x)
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        x = torch.cat((x, cls_emb), 1)
        for i, blk in enumerate(self.blocks):
            if i < layer_id:
                x = blk(x)
            else:
                return blk(x, return_attention=True)

def padding(im, patch_size, fill_value=0):
    # make the image sizes divisible by patch_size
    H, W = im.size(2), im.size(3)
    pad_h, pad_w = 0, 0
    if H % patch_size > 0:
        pad_h = patch_size - (H % patch_size)
    if W % patch_size > 0:
        pad_w = patch_size - (W % patch_size)
    im_padded = im
    if pad_h > 0 or pad_w > 0:
        im_padded = F.pad(im, (0, pad_w, 0, pad_h), value=fill_value)
    return im_padded


def unpadding(y, target_size):
    H, W = target_size
    H_pad, W_pad = y.size(2), y.size(3)
    # crop predictions on extra pixels coming from padding
    extra_h = H_pad - H
    extra_w = W_pad - W
    if extra_h > 0:
        y = y[:, :, :-extra_h]
    if extra_w > 0:
        y = y[:, :, :, :-extra_w]
    return y


class Segmenter(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        n_cls,
    ):
        super().__init__()
        self.n_cls = n_cls
        self.patch_size = encoder.patch_embed.patch_size[0]
        self.encoder = encoder
        self.decoder = decoder

    # @torch.jit.ignore
    # def no_weight_decay(self):
    #     def append_prefix_no_weight_decay(prefix, module):
    #         return set(map(lambda x: prefix + x, module.no_weight_decay()))
    #
    #     nwd_params = append_prefix_no_weight_decay("encoder.", self.encoder).union(
    #         append_prefix_no_weight_decay("decoder.", self.decoder)
    #     )
    #     return nwd_params

    def forward(self, im):
        B, H_ori, W_ori = im.size(0), im.size(2), im.size(3)
        im = padding(im, self.patch_size)
        H, W = im.size(2), im.size(3)

        x = self.encoder(im)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens =self.encoder.num_prefix_tokens
        x = x[:, num_extra_tokens:]

        masks = self.decoder(x, (H, W))

        masks = F.interpolate(masks, size=(H, W), mode="bilinear")
        masks = unpadding(masks, (H_ori, W_ori))

        return masks

    def get_attention_map_enc(self, im, layer_id):
        return self.encoder.get_attention_map(im, layer_id)

    def get_attention_map_dec(self, im, layer_id):
        x = self.encoder(im, return_features=True)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]

        return self.decoder.get_attention_map(x, layer_id)

class MySegmenter(nn.Module):
    def __init__(
        self,
        fusion_model,
        decoder,
        n_cls,
    ):
        super().__init__()
        self.n_cls = n_cls
        self.encoder = fusion_model.img_encoder
        self.patch_size = self.encoder.patch_embed.patch_size[0]
        self.img_norm = fusion_model.img_norm
        self.decoder = decoder

        self.img_pos_embed=fusion_model.img_pos_embed
        self.cls_token=fusion_model.cls_token


    # @torch.jit.ignore
    # def no_weight_decay(self):
    #     def append_prefix_no_weight_decay(prefix, module):
    #         return set(map(lambda x: prefix + x, module.no_weight_decay()))
    #
    #     nwd_params = append_prefix_no_weight_decay("encoder.", self.encoder).union(
    #         append_prefix_no_weight_decay("decoder.", self.decoder)
    #     )
    #     return nwd_params

    def forward(self, im):
        B, H_ori, W_ori = im.size(0), im.size(2), im.size(3)
        im = padding(im, self.patch_size)
        H, W = im.size(2), im.size(3)

        # embed patches
        x = self.encoder.patch_embed(im)
        # add pos embed w/o cls token
        x = x + self.img_pos_embed[:, 1:, :]
        cls_token = self.cls_token + self.img_pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.encoder.blocks:
            x = blk(x)
        x = self.img_norm(x)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens =self.encoder.num_prefix_tokens
        x = x[:, num_extra_tokens:]

        masks = self.decoder(x, (H, W))

        masks = F.interpolate(masks, size=(H, W), mode="bilinear")
        masks = unpadding(masks, (H_ori, W_ori))

        return masks

    def get_attention_map_enc(self, im, layer_id):
        return self.encoder.get_attention_map(im, layer_id)

    def get_attention_map_dec(self, im, layer_id):
        x = self.encoder(im, return_features=True)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]

        return self.decoder.get_attention_map(x, layer_id)


class MyFusionSegmenter(nn.Module):
    def __init__(
        self,
        fusion_model,
        decoder,
        n_cls,
    ):
        super().__init__()
        self.n_cls = n_cls
        self.img_encoder = fusion_model.img_encoder
        self.osm_encoder = fusion_model.osm_encoder
        self.img_mlp = fusion_model.img_decoder_mlp
        self.osm_mlp = fusion_model.osm_decoder_mlp
        self.img_norm = fusion_model.img_norm
        self.osm_norm = fusion_model.osm_norm
        self.fusion_blocks=fusion_model.img_cross_modal_encoder
        self.patch_size = self.img_encoder.patch_embed.patch_size[0]
        self.img_pos_embed = fusion_model.img_pos_embed
        self.cls_token = fusion_model.cls_token
        self.cross_img_pos_embed=fusion_model.cross_img_pos_embed
        self.osm_read_out = fusion_model.osm_read_out

        self.decoder = decoder

    def forward_img_encoder(self, x):
        # embed patches
        x = self.img_encoder.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.img_pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.img_pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.img_encoder.blocks:
            x = blk(x)
        x = self.img_norm(x)
        x = self.img_mlp(x)

        return x

    def forward_osm_encoder(self, osms, node_pos):
        device = self.img_pos_embed.device
        use_osms=osms.clone()

        node_pos = node_pos * (self.img_encoder.patch_embed.num_patches ** .5) # relative pos to absolute pos
        node_pos = node_pos.transpose(1,0)
        node_pos_embedding = get_2d_sincos_pos_embed_from_grid(use_osms.x.shape[-1], grid=node_pos)
        node_pos_embedding = torch.from_numpy(node_pos_embedding).to(device).float()
        use_osms.x = use_osms.x + node_pos_embedding

        out = self.osm_encoder(use_osms)
        out = self.osm_norm(out)

        out = self.osm_read_out(out, osms.batch.long()).float()
        out = out.reshape(-1, 1, out.shape[-1])
        out = self.osm_mlp(out)

        return out

    def forward(self, im, osms):
        node_pos = osms.x[:, -3:-1].clone().detach().cpu().numpy()
        osms.x = osms.x[:, :-1]

        B, H_ori, W_ori = im.size(0), im.size(2), im.size(3)
        im = padding(im, self.patch_size)
        H, W = im.size(2), im.size(3)

        img_embedding=self.forward_img_encoder(im)
        osm_embedding=self.forward_osm_encoder(osms, node_pos)

        for blk in self.fusion_blocks:
            osm_embedding, img_embedding = blk(osm_embedding, img_embedding, osm_embedding, self.cross_img_pos_embed)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens =self.img_encoder.num_prefix_tokens
        img_embedding = img_embedding[:, num_extra_tokens:]

        masks = self.decoder(img_embedding, (H, W))

        masks = F.interpolate(masks, size=(H, W), mode="bilinear")
        masks = unpadding(masks, (H_ori, W_ori))

        return masks

    def get_attention_map_enc(self, im, layer_id):
        return self.encoder.get_attention_map(im, layer_id)

    def get_attention_map_dec(self, im, layer_id):
        x = self.encoder(im, return_features=True)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]

        return self.decoder.get_attention_map(x, layer_id)


class MyFineFusionSegmenter(nn.Module):
    def __init__(
            self,
            fusion_model,
            decoder,
            n_cls,
    ):
        super().__init__()
        self.n_cls = n_cls
        self.img_encoder = fusion_model.img_encoder
        self.osm_encoder = fusion_model.osm_encoder
        self.img_mlp = fusion_model.img_decoder_mlp
        self.osm_node_decoder_mlp=fusion_model.osm_node_decoder_mlp

        self.img_norm = fusion_model.img_norm
        self.osm_norm = fusion_model.osm_norm
        self.fusion_blocks = fusion_model.osm_cross_modal_encoder
        self.patch_size = self.img_encoder.patch_embed.patch_size[0]
        self.img_pos_embed = fusion_model.img_pos_embed
        self.cls_token = fusion_model.cls_token
        self.cross_img_pos_embed = fusion_model.cross_img_pos_embed
        self.num_patches=self.img_encoder.patch_embed.num_patches

        self.decoder_embed_dim=self.img_encoder.embed_dim

        self.decoder = decoder

    def forward_img_encoder(self, x):
        # embed patches
        x = self.img_encoder.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.img_pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.img_pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.img_encoder.blocks:
            x = blk(x)
        x = self.img_norm(x)
        x = self.img_mlp(x)

        return x

    def forward_osm_encoder(self, osms, node_pos):
        device = self.img_pos_embed.device
        use_osms = osms.clone()

        node_pos = node_pos * (self.img_encoder.patch_embed.num_patches ** .5)  # relative pos to absolute pos
        node_pos = node_pos.transpose(1, 0)
        node_pos_embedding = get_2d_sincos_pos_embed_from_grid(use_osms.x.shape[-1], grid=node_pos)
        node_pos_embedding = torch.from_numpy(node_pos_embedding).to(device).float()
        use_osms.x = use_osms.x + node_pos_embedding

        out = self.osm_encoder(use_osms)
        out = self.osm_norm(out)

        return out

    # def forward_cross_encoder(self, osms, osm_node_embedding, img_embedding, node_pos):
    #     device = self.img_pos_embed.device
    #     batch_size=img_embedding.shape[0]
    #     batch_index = osms.batch
    #
    #     osm_node_embedding = self.osm_node_decoder_mlp(osm_node_embedding)
    #     node_pos = node_pos * (self.num_patches ** .5)  # relative pos to absolute pos
    #     node_pos = node_pos.transpose(1, 0)
    #     node_pos_embedding = get_2d_sincos_pos_embed_from_grid(self.decoder_embed_dim, grid=node_pos)
    #     node_pos_embedding = torch.from_numpy(node_pos_embedding).to(device).reshape(-1, 1, node_pos_embedding.shape[
    #         -1]).float()
    #     osm_node_embedding = osm_node_embedding.reshape(-1, 1, osm_node_embedding.shape[-1])
    #
    #
    #     # repeat img mask embeddings
    #     img_embedding = self.img_mlp(img_embedding)
    #
    #     img_embedding = img_embedding[batch_index]
    #     # print(img_embedding.shape, osm_mask_embedding.shape, node_pos_embedding.shape)
    #
    #     # apply Cross-modal Transformer blocks
    #     for blk in self.fusion_blocks:
    #         img_embedding, osm_mask_embedding = blk(img_embedding, osm_node_embedding, self.cross_img_pos_embed,
    #                                                 node_pos_embedding)
    #
    #     # Initialize tensors to accumulate sums and counts
    #     sums = torch.zeros(batch_size, img_embedding.size(1), img_embedding.size(2), dtype=img_embedding.dtype).to(
    #             device)
    #     counts = torch.zeros(batch_size, dtype=batch_index.dtype).to(device)
    #
    #     # Accumulate sums and counts using scatter_add_
    #     sums.scatter_add_(0, batch_index.unsqueeze(1).unsqueeze(2).expand(-1, img_embedding.size(1),
    #                                                                           img_embedding.size(2)), img_embedding)
    #     counts.scatter_add_(0, batch_index, torch.ones_like(batch_index))
    #
    #     img_embedding = sums / counts.unsqueeze(1).unsqueeze(2).to(torch.float)
    #     return img_embedding

    def forward_cross_encoder(self, osms, osm_node_embedding, img_embedding, node_pos, sigma=0.5):
        device = self.img_pos_embed.device
        batch_size=img_embedding.shape[0]
        batch_index = osms.batch

        osm_node_embedding = self.osm_node_decoder_mlp(osm_node_embedding)
        node_pos = node_pos * (self.num_patches ** .5)  # relative pos to absolute pos
        node_pos = node_pos.transpose(1, 0)
        node_pos_embedding = get_2d_sincos_pos_embed_from_grid(self.decoder_embed_dim, grid=node_pos)
        node_pos_embedding = torch.from_numpy(node_pos_embedding).to(device).reshape(-1, 1, node_pos_embedding.shape[
            -1]).float()
        osm_node_embedding = osm_node_embedding.reshape(-1, 1, osm_node_embedding.shape[-1])


        # repeat img mask embeddings
        img_embedding = self.img_mlp(img_embedding)

        img_embedding = img_embedding[batch_index]
        # print(img_embedding.shape, osm_mask_embedding.shape, node_pos_embedding.shape)

        # apply Cross-modal Transformer blocks
        for blk in self.fusion_blocks:
            img_embedding, osm_mask_embedding = blk(img_embedding, osm_node_embedding, self.cross_img_pos_embed[:, 1:,:],
                                                    node_pos_embedding)

        cur_img_pos_embed=torch.from_numpy(get_2d_sincos_pos_embed(self.cross_img_pos_embed.shape[-1],
                                                      int(self.num_patches ** .5), cls_token=False)).float().to(device)
        w = (torch.squeeze(node_pos_embedding) @ torch.squeeze(cur_img_pos_embed).T)
        w_min = w.min(dim=1)[0].reshape(-1, 1)
        w_max = w.max(dim=1)[0].reshape(-1, 1)
        w = (w - w_min) / (w_max - w_min)
        w = torch.exp(w / sigma)

        img_embedding = img_embedding * w.unsqueeze(-1)

        img_sums = torch.zeros(batch_size, img_embedding.size(1), img_embedding.size(2), dtype=img_embedding.dtype).to(device)

        img_sums.scatter_add_(0, batch_index.unsqueeze(1).unsqueeze(2).expand(-1, img_embedding.size(1),
                                                                                  img_embedding.size(2)), img_embedding)

        img_counts = torch.zeros(batch_size, w.size(1), dtype=w.dtype).to(device)

        img_counts.scatter_add_(0, batch_index.unsqueeze(1).expand(-1, w.size(1)), w)
        img_embedding = img_sums / img_counts.unsqueeze(-1)

        return img_embedding

    def forward(self, im, osms):
        node_pos = osms.x[:, -3:-1].clone().detach().cpu().numpy()
        osms.x = osms.x[:, :-1]

        B, H_ori, W_ori = im.size(0), im.size(2), im.size(3)
        im = padding(im, self.patch_size)
        H, W = im.size(2), im.size(3)

        img_embedding = self.forward_img_encoder(im)
        osm_embedding = self.forward_osm_encoder(osms, node_pos)

        # img_embedding = self.forward_cross_encoder(osms, osm_embedding, img_embedding, node_pos)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = self.img_encoder.num_prefix_tokens
        img_embedding = img_embedding[:, num_extra_tokens:]

        img_embedding = self.forward_cross_encoder(osms, osm_embedding, img_embedding, node_pos)

        masks = self.decoder(img_embedding, (H, W))

        masks = F.interpolate(masks, size=(H, W), mode="bilinear")
        masks = unpadding(masks, (H_ori, W_ori))

        return masks

    def get_attention_map_enc(self, im, layer_id):
        return self.encoder.get_attention_map(im, layer_id)

    def get_attention_map_dec(self, im, layer_id):
        x = self.encoder(im, return_features=True)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]

        return self.decoder.get_attention_map(x, layer_id)

def create_decoder(embed_dim, patch_size, decoder_cfg):
    decoder_cfg = decoder_cfg.copy()
    name = decoder_cfg.pop("name")
    # decoder_cfg["d_encoder"] = encoder.embed_dim
    # decoder_cfg["patch_size"] = encoder.patch_embed.patch_size[0]
    decoder_cfg["d_encoder"] = embed_dim
    decoder_cfg["patch_size"] = patch_size

    if "linear" in name:
        decoder = DecoderLinear(**decoder_cfg)
    elif name == "mask_transformer":
        dim = embed_dim
        n_heads = dim // 64
        decoder_cfg["n_heads"] = n_heads
        decoder_cfg["d_model"] = dim
        decoder_cfg["d_ff"] = 4 * dim
        decoder = MaskTransformer(**decoder_cfg)
    else:
        raise ValueError(f"Unknown decoder: {name}")
    return decoder
