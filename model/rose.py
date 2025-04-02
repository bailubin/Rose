# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
import random
import torch
import torch.nn as nn
import torch_geometric
import torch.nn.functional as F

from timm.models.vision_transformer import PatchEmbed, Block

from .pos_embed import get_2d_sincos_pos_embed, get_2d_sincos_pos_embed_from_grid, get_1d_sincos_pos_embed_from_grid
from .transformer import TwoWayAttentionBlock

class OsmEncoder(torch.nn.Module):
    def __init__(self, osm_in_chans, osm_out_dim):
        super(OsmEncoder, self).__init__()

        # self.gcn1 = torch_geometric.nn.GCNConv(in_channels=osm_in_chans, out_channels=int(osm_out_dim * 2),
        #                                        add_self_loops=True)
        # self.norm = torch_geometric.nn.LayerNorm(int(osm_out_dim * 2))
        # self.gcn2 = torch_geometric.nn.GCNConv(in_channels=int(osm_out_dim * 2), out_channels=osm_out_dim,
        #                                        add_self_loops=True)

        # self.gcn1 = torch_geometric.nn.GCNConv(in_channels=osm_in_chans, out_channels=osm_out_dim,
        #                                        add_self_loops=True)
        self.gcn1 = torch_geometric.nn.GATConv(in_channels=osm_in_chans, out_channels=osm_out_dim, add_self_loops=True)

    def forward(self, data):
        x, edge_index, edge_weight = data.x.float(), data.edge_index.long(), data.edge_attr

        out = self.gcn1(x, edge_index, edge_weight)
        # out = self.norm(out)
        # out = self.gcn2(out, edge_index, edge_weight)
        return out

class Rose(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_encoder, osm_encoder, img_in_chans=3, osm_in_chans=524, osm_out_chans=10,
                 osm_out_embed_dim=128, cross_attention_downsample_rate=2,
                 img_cross_modal_encoder_depth=1, img_cross_modal_encoder_num_heads=8,
                 img_cross_modal_decoder_depth=1, img_cross_modal_decoder_num_heads=8,
                 osm_cross_modal_encoder_depth=1, osm_cross_modal_encoder_num_heads=8,
                 decoder_embed_dim=512, mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # Image encoder specifics
        self.img_encoder = img_encoder
        self.patch_embed = self.img_encoder.patch_embed
        num_patches = self.patch_embed.num_patches
        patch_size = self.patch_embed.patch_size[0]
        img_embed_dim = self.img_encoder.embed_dim
        self.grid_size=int(num_patches ** .5)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, img_embed_dim))
        self.img_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, img_embed_dim), requires_grad=False)
        self.img_norm = norm_layer(img_embed_dim)

        self.osm_encoder = osm_encoder
        self.osm_norm = torch_geometric.nn.LayerNorm(osm_out_embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # Image fusion encoder and decoder specifics
        decoder_embed_dim = img_embed_dim

        self.cross_img_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),requires_grad=False)  # fixed sin-cos embedding

        self.img_decoder_mlp = nn.Linear(img_embed_dim, decoder_embed_dim, bias=True)
        self.osm_read_out = torch_geometric.nn.Set2Set(osm_out_embed_dim, processing_steps=5)
        self.osm_decoder_mlp = nn.Linear(int(osm_out_embed_dim*2), decoder_embed_dim, bias=True)

        self.img_cross_modal_encoder = nn.ModuleList([
            TwoWayAttentionBlock(
                embedding_dim=decoder_embed_dim,
                num_heads=img_cross_modal_encoder_num_heads,
                attention_downsample_rate=cross_attention_downsample_rate)
            for i in range(img_cross_modal_encoder_depth)
        ])

        self.img_mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.img_cross_modal_decoder = nn.ModuleList([
            Block(decoder_embed_dim, img_cross_modal_decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(img_cross_modal_decoder_depth)])
        # self.img_cross_modal_decoder = nn.ModuleList([
        #     TwoWayAttentionBlock(
        #         embedding_dim=decoder_embed_dim,
        #         num_heads=img_cross_modal_decoder_num_heads,
        #         attention_downsample_rate=cross_attention_downsample_rate)
        #     for i in range(img_cross_modal_decoder_depth)
        # ])


        self.img_decoder_norm = norm_layer(decoder_embed_dim)
        self.img_decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * img_in_chans, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # Osm fusion encoder and decoder specifics
        self.decoder_embed_dim=decoder_embed_dim
        self.osm_node_decoder_mlp = nn.Linear(osm_out_embed_dim, decoder_embed_dim, bias=True)
        self.osm_mask_token = torch.nn.Parameter(torch.zeros(1, osm_in_chans))
        self.osm_cross_modal_encoder = nn.ModuleList([
            TwoWayAttentionBlock(
                embedding_dim=decoder_embed_dim,
                num_heads=osm_cross_modal_encoder_num_heads,
                attention_downsample_rate=cross_attention_downsample_rate)
            for i in range(osm_cross_modal_encoder_depth)
        ])

        self.osm_decoder_norm = norm_layer(decoder_embed_dim)
        self.osm_decoder_pred = nn.Linear(decoder_embed_dim, osm_in_chans,
                                          bias=True)
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        img_pos_embed = get_2d_sincos_pos_embed(self.img_pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                                cls_token=True)
        self.img_pos_embed.data.copy_(torch.from_numpy(img_pos_embed).float().unsqueeze(0))
        # self.img_pos_embed=torch.from_numpy(img_pos_embed).float().unsqueeze(0)

        cross_img_pos_embed = get_2d_sincos_pos_embed(self.cross_img_pos_embed.shape[-1],
                                                      int(self.patch_embed.num_patches ** .5), cls_token=True)
        self.cross_img_pos_embed.data.copy_(torch.from_numpy(cross_img_pos_embed).float().unsqueeze(0))
        # self.cross_img_pos_embed=torch.from_numpy(cross_img_pos_embed).float().unsqueeze(0)

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.img_encoder.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.img_mask_token, std=.02)
        torch.nn.init.normal_(self.osm_mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        # self.apply(self._init_weights)
        for name,module in self.named_children():
            if 'img_encoder' not in name:
                module.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_mask_img(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def random_mask_node(self, g, device, mask_ratio):
        num_nodes = g.x.shape[0]
        perm = torch.randperm(num_nodes, device=device)

        num_mask_nodes = int(mask_ratio * num_nodes + 0.5)
        if num_mask_nodes == 0:
            num_mask_nodes = 1
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        out_g = g.clone()
        out_g.x[mask_nodes] = 0.0
        out_g.x[mask_nodes] += self.osm_mask_token

        return out_g, mask_nodes, keep_nodes

    def forward_img_encoder(self, x, mask_ratio):
        # embed patches
        x = self.img_encoder.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.img_pos_embed[:, 1:, :]

        if mask_ratio > 0:
            # masking: length -> length * mask_ratio
            x, mask, ids_restore = self.random_mask_img(x, mask_ratio)
        else:
            mask, ids_restore = None, None

        # append cls token
        cls_token = self.cls_token + self.img_pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.img_encoder.blocks:
            x = blk(x)
        x = self.img_norm(x)

        return x, mask, ids_restore

    def forward_osm_encoder(self, osms, node_pos, mask_ratio, drop_edge_rate=0):
        device = self.img_pos_embed.device
        use_osms=osms.clone()

        if mask_ratio > 0:
            use_osms, mask_nodes, keep_nodes = self.random_mask_node(use_osms, device, mask_ratio)
        else:
            mask_nodes, keep_nodes=None, None

        if drop_edge_rate > 0:
            use_edge_index, edge_mask = torch_geometric.utils.dropout_edge(use_osms.edge_index, drop_edge_rate)
            use_osms.edge_index = use_edge_index
            use_osms.edge_attr = use_osms.edge_attr[edge_mask]

        node_pos = node_pos * (self.patch_embed.num_patches ** .5) # relative pos to absolute pos
        node_pos = node_pos.transpose(1,0)
        node_pos_embedding = get_2d_sincos_pos_embed_from_grid(use_osms.x.shape[-1], grid=node_pos)
        node_pos_embedding = torch.from_numpy(node_pos_embedding).to(device).float()

        # node_pos_embedding = torch.zeros((use_osms.x.shape[0], use_osms.x.shape[-1])).to(device).float()

        use_osms.x = use_osms.x + node_pos_embedding

        out = self.osm_encoder(use_osms)
        out = self.osm_norm(out)

        return out, mask_nodes, keep_nodes

    def forward_img_cross_encoder_decoder(self, img_embedding, osm_embedding, ids_restore):
        # embed tokens
        img_embedding = self.img_decoder_mlp(img_embedding)
        osm_embedding = self.osm_decoder_mlp(osm_embedding)

        # append mask tokens to sequence
        mask_tokens = self.img_mask_token.repeat(img_embedding.shape[0], ids_restore.shape[1] + 1 - img_embedding.shape[1], 1)
        x_ = torch.cat([img_embedding[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, img_embedding.shape[2]))  # unshuffle
        img_embedding = torch.cat([img_embedding[:, :1, :], x_], dim=1)  # append cls token

        # apply Cross-modal Transformer blocks
        for blk in self.img_cross_modal_encoder:
            osm_embedding, img_embedding = blk(osm_embedding, img_embedding, osm_embedding, self.cross_img_pos_embed)

        # apply Decoder Transformer blocks
        img_embedding = img_embedding + self.cross_img_pos_embed
        for blk in self.img_cross_modal_decoder:
            img_embedding = blk(img_embedding)
        # for blk in self.img_cross_modal_decoder:
        #     osm_embedding, img_embedding = blk(osm_embedding, img_embedding, osm_embedding, self.cross_img_pos_embed)


        img_embedding = self.img_decoder_norm(img_embedding)

        # predictor projection
        out = self.img_decoder_pred(img_embedding)

        # remove cls token
        out = out[:, 1:, :]

        return out

    def forward_osm_cross_encoder_decoder(self, osms, osm_node_embeddings, img_embedding, mask_nodes, node_pos):
        device = self.img_pos_embed.device

        node_pos = node_pos[mask_nodes.cpu().numpy(), -2:]
        node_pos = node_pos * (self.patch_embed.num_patches ** .5)  # relative pos to absolute pos
        node_pos = node_pos.transpose(1, 0)
        node_pos_embedding = get_2d_sincos_pos_embed_from_grid(self.decoder_embed_dim, grid=node_pos)
        node_pos_embedding = torch.from_numpy(node_pos_embedding).to(device).reshape(-1,1,node_pos_embedding.shape[-1]).float()

        # get osm mask embeddings
        osm_mask_embedding = osm_node_embeddings[mask_nodes]
        osm_mask_embedding = osm_mask_embedding.reshape(-1, 1, osm_mask_embedding.shape[-1])
        osm_mask_embedding = self.osm_node_decoder_mlp(osm_mask_embedding)

        # repeat img mask embeddings
        img_embedding = self.img_decoder_mlp(img_embedding)
        mask_batch_index = osms.batch[mask_nodes]
        img_embedding = img_embedding[mask_batch_index]
        # print(img_embedding.shape, osm_mask_embedding.shape, node_pos_embedding.shape)

        # apply Cross-modal Transformer blocks
        for blk in self.osm_cross_modal_encoder:
            img_embedding, osm_mask_embedding = blk(img_embedding, osm_mask_embedding, self.cross_img_pos_embed,
                                                    node_pos_embedding)

        osm_mask_embedding = self.osm_decoder_norm(osm_mask_embedding)

        # predictor projection
        osm_mask_embedding = torch.squeeze(osm_mask_embedding)
        out = self.osm_decoder_pred(osm_mask_embedding)
        return out

    def forward_img_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward_geo_loss(self, osms, pred, alpha=3):
        # x = F.normalize(osms, p=2, dim=-1)
        # y = F.normalize(pred, p=2, dim=-1)
        #
        # loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
        #
        # loss = loss.mean()
        loss=(osms-pred)**2
        loss=loss.mean(dim=-1).mean()

        return loss


    def forward(self, imgs, osms, img_mask_ratio=0.75, osm_mask_ratio=0.1):
        node_label = osms.x[:, -1].long()
        node_pos = osms.x[:, -3:-1].clone().detach().cpu().numpy()
        osms.x = osms.x[:, :-1]

        # reconstruct img
        # get img embedding
        img_embedding, mask_imgs, ids_restore = self.forward_img_encoder(imgs, mask_ratio=img_mask_ratio)

        # get osm embedding
        osm_node_embeddings, _, _ = self.forward_osm_encoder(osms, node_pos, mask_ratio=0, drop_edge_rate=0)
        osm_embedding = self.osm_read_out(osm_node_embeddings, osms.batch.long()).float()
        osm_embedding = osm_embedding.reshape(-1, 1, osm_embedding.shape[-1])

        pred_imgs = self.forward_img_cross_encoder_decoder(img_embedding, osm_embedding, ids_restore)  # [N, L, p*p*3]
        img_loss = self.forward_img_loss(imgs, pred_imgs, mask_imgs)

        # reconstruct osm
        # get img embedding
        img_embedding, _, _ = self.forward_img_encoder(imgs, mask_ratio=0)

        # get osm embedding
        osm_node_embeddings, mask_nodes, keep_nodes = self.forward_osm_encoder(osms, node_pos, mask_ratio=osm_mask_ratio)

        pred_osm = self.forward_osm_cross_encoder_decoder(osms, osm_node_embeddings, img_embedding, mask_nodes, node_pos)
        osm_loss = self.forward_geo_loss(osms.x[mask_nodes], pred_osm)


        return img_loss, osm_loss, pred_imgs, mask_imgs, pred_osm, mask_nodes

        # return img_loss, osm_label_loss, osm_geo_loss, pred_imgs, mask_imgs, pred_geo, mask_nodes

    def get_osm_to_img_attn(self, imgs, osms, mask_node_inds):
        node_pos = osms.x[:, -3:-1].clone().detach().cpu().numpy()
        osms.x = osms.x[:, :-1]

        use_g = osms.clone()
        use_g.x[mask_node_inds] = 0.0
        use_g.x[mask_node_inds] += self.osm_mask_token

        img_embedding, _, _ = self.forward_img_encoder(imgs, mask_ratio=0)
        osm_node_embeddings, _, _ = self.forward_osm_encoder(osms, node_pos, mask_ratio=0)

        device = self.img_pos_embed.device

        node_pos = node_pos[mask_node_inds, -2:]
        node_pos = node_pos * (self.patch_embed.num_patches ** .5)  # relative pos to absolute pos
        node_pos = node_pos.transpose(1, 0)
        node_pos_embedding = get_2d_sincos_pos_embed_from_grid(self.decoder_embed_dim, grid=node_pos)
        node_pos_embedding = torch.from_numpy(node_pos_embedding).to(device).reshape(-1, 1, node_pos_embedding.shape[
            -1]).float()
        # get osm mask embeddings
        osm_mask_embedding = osm_node_embeddings[mask_node_inds]
        osm_mask_embedding = osm_mask_embedding.reshape(-1, 1, osm_mask_embedding.shape[-1])
        osm_mask_embedding = self.osm_node_decoder_mlp(osm_mask_embedding)

        # repeat img mask embeddings
        img_embedding = self.img_decoder_mlp(img_embedding)
        img_embedding = img_embedding.repeat(osm_mask_embedding.shape[0], 1, 1)
        # print(img_embedding.shape, osm_mask_embedding.shape, node_pos_embedding.shape)

        # apply Cross-modal Transformer blocks
        attns=[]
        for blk in self.osm_cross_modal_encoder:
            img_embedding, osm_mask_embedding, attn = blk(img_embedding, osm_mask_embedding, self.cross_img_pos_embed,
                                                    node_pos_embedding, require_attn=True)
            # b, n_heads, n_tokens, c_per_head = attn.shape
            # attn = attn.transpose(1, 2)
            # attn = attn.reshape(b, n_tokens, n_heads * c_per_head)
            attns.append(attn)
        return attns

    def get_osm_to_img_attn_nopos(self, imgs, osms, mask_node_inds):
        node_pos = osms.x[:, -3:-1].clone().detach().cpu().numpy()
        osms.x = osms.x[:, :-1]

        use_g = osms.clone()
        use_g.x[mask_node_inds] = 0.0
        use_g.x[mask_node_inds] += self.osm_mask_token

        img_embedding, _, _ = self.forward_img_encoder(imgs, mask_ratio=0)
        osm_node_embeddings, _, _ = self.forward_osm_encoder(osms, node_pos, mask_ratio=0)

        device = self.img_pos_embed.device

        # node_pos = node_pos[mask_node_inds, -2:]
        # node_pos = node_pos * (self.patch_embed.num_patches ** .5)  # relative pos to absolute pos
        # node_pos = node_pos.transpose(1, 0)
        # node_pos_embedding = get_2d_sincos_pos_embed_from_grid(self.decoder_embed_dim, grid=node_pos)
        # node_pos_embedding = torch.from_numpy(node_pos_embedding).to(device).reshape(-1, 1, node_pos_embedding.shape[
        #     -1]).float()
        node_pos_embedding = torch.zeros((node_pos.shape[-1], self.decoder_embed_dim)).to(device)
        node_pos_embedding = node_pos_embedding.reshape(-1,1,node_pos_embedding.shape[-1]).float()
        # get osm mask embeddings
        osm_mask_embedding = osm_node_embeddings[mask_node_inds]
        osm_mask_embedding = osm_mask_embedding.reshape(-1, 1, osm_mask_embedding.shape[-1])
        osm_mask_embedding = self.osm_node_decoder_mlp(osm_mask_embedding)

        # repeat img mask embeddings
        img_embedding = self.img_decoder_mlp(img_embedding)
        img_embedding = img_embedding.repeat(osm_mask_embedding.shape[0], 1, 1)
        # print(img_embedding.shape, osm_mask_embedding.shape, node_pos_embedding.shape)

        # apply Cross-modal Transformer blocks
        attns=[]
        for blk in self.osm_cross_modal_encoder:
            img_embedding, osm_mask_embedding, attn = blk(img_embedding, osm_mask_embedding, 0.,
                                                    0., require_attn=True)
            # b, n_heads, n_tokens, c_per_head = attn.shape
            # attn = attn.transpose(1, 2)
            # attn = attn.reshape(b, n_tokens, n_heads * c_per_head)
            attns.append(attn)
        return attns