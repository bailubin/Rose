import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
import torch
from einops import rearrange
import torch.nn.functional as F
from .pos_embed import get_2d_sincos_pos_embed, get_2d_sincos_pos_embed_from_grid, get_1d_sincos_pos_embed_from_grid

class PopLinear(nn.Module):
    def __init__(
        self,
        fusion_model
    ):
        super().__init__()
        self.encoder = fusion_model.img_encoder
        self.patch_size = self.encoder.patch_embed.patch_size[0]
        self.img_norm = fusion_model.img_norm

        self.img_pos_embed=fusion_model.img_pos_embed
        self.cls_token=fusion_model.cls_token

        self.decoder=nn.Linear(self.encoder.embed_dim, 1, bias=True)
        nn.init.constant_(self.decoder.bias, 0)
        nn.init.constant_(self.decoder.weight, 1.0)
        # dim_mlp = self.encoder.embed_dim
        # self.decoder = nn.Sequential(nn.Linear(dim_mlp, int(dim_mlp / 2), bias=True),
        #                             nn.ReLU(),
        #                             nn.Linear(int(dim_mlp / 2), 1, bias=True))

    def forward(self, im):

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

        x = x[:, 0]

        x = F.relu(self.decoder(x))

        return x

class PopFusionLinear(nn.Module):
    def __init__(
        self,
        fusion_model,
        cross_dim=512
    ):
        super().__init__()
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

        self.decoder = nn.Linear(cross_dim, 1, bias=True)
        nn.init.constant_(self.decoder.bias, 0)
        nn.init.constant_(self.decoder.weight, 1.0)

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

        img_embedding=self.forward_img_encoder(im)
        osm_embedding=self.forward_osm_encoder(osms, node_pos)

        for blk in self.fusion_blocks:
            osm_embedding, img_embedding = blk(osm_embedding, img_embedding, osm_embedding, self.cross_img_pos_embed)

        x = img_embedding[:, 0]

        x = F.relu(self.decoder(x))

        return x

class PopOSMLinear(nn.Module):
    def __init__(
        self,
        fusion_model,
        cross_dim=512
    ):
        super().__init__()
        self.osm_encoder = fusion_model.osm_encoder
        # self.osm_mlp = fusion_model.osm_decoder_mlp
        self.osm_norm = fusion_model.osm_norm
        self.osm_read_out = fusion_model.osm_read_out

        self.num_patches = fusion_model.img_encoder.patch_embed.num_patches
        self.img_pos_embed = fusion_model.img_pos_embed

        self.decoder = nn.Linear(cross_dim, 1, bias=True)
        nn.init.constant_(self.decoder.bias, 0)
        nn.init.constant_(self.decoder.weight, 1.0)


    def forward_osm_encoder(self, osms, node_pos):
        device = self.img_pos_embed.device
        use_osms=osms.clone()

        node_pos = node_pos * (self.num_patches ** .5) # relative pos to absolute pos
        node_pos = node_pos.transpose(1,0)
        node_pos_embedding = get_2d_sincos_pos_embed_from_grid(use_osms.x.shape[-1], grid=node_pos)
        node_pos_embedding = torch.from_numpy(node_pos_embedding).to(device).float()
        use_osms.x = use_osms.x + node_pos_embedding

        out = self.osm_encoder(use_osms)
        out = self.osm_norm(out)

        out = self.osm_read_out(out, osms.batch.long()).float()
        # out = self.osm_mlp(out)
        # out = out.reshape(-1, 1, out.shape[-1])
        # out = self.osm_mlp(out)

        return out

    def forward(self, osms):
        node_pos = osms.x[:, -3:-1].clone().detach().cpu().numpy()
        osms.x = osms.x[:, :-1]

        osm_embedding=self.forward_osm_encoder(osms, node_pos)

        x = F.relu(self.decoder(osm_embedding))

        return x