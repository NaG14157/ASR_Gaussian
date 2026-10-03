import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from einops import rearrange
#from torchinfo import summary

# for HCI
class Net_CrossAttention(nn.Module):
    def __init__(self, angular_in, angular_out, use_volume_template=True, max_offset_px=12.0,
                 use_adaptive_offset=True, adaptive_offset_min_px=8.0, adaptive_offset_max_px=16.0,
                 use_confidence_gate=True, confidence_gate_blend=1.0, confidence_temperature=0.7):
        super(Net_CrossAttention, self).__init__()
        channel = 64
        self.channel = channel
        self.angRes = angular_in
        self.angRes_out = angular_out
        self.feature_extractor = ASRFeatureExtractor(self.angRes, channel)
        self.epiFeatureRebuild = FourVolumeFeatureRebuild(
            self.angRes,
            self.angRes_out,
            channel,
            feat_unfold=False,
            use_volume_template=use_volume_template,
            max_offset_px=max_offset_px,
            use_adaptive_offset=use_adaptive_offset,
            adaptive_offset_min_px=adaptive_offset_min_px,
            adaptive_offset_max_px=adaptive_offset_max_px,
            use_confidence_gate=use_confidence_gate,
            confidence_gate_blend=confidence_gate_blend,
            confidence_temperature=confidence_temperature
        )
        self.DownSample = nn.Sequential(
            nn.Conv3d(channel, channel // 4, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(channel // 4 , channel // 4 // 4, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(channel // 4 // 4, 1, kernel_size=1, stride=1,
                      padding=0, bias=False),
        )
        # self.UpSample = nn.Sequential(
        #     nn.Conv3d(channel * angular_in * angular_in, channel // 4 * angular_in * angular_in, kernel_size=1,
        #               stride=1, padding=0, bias=False),
        #     nn.LeakyReLU(0.1, inplace=True),
        #     nn.Conv3d(channel // 4 * angular_in * angular_in, channel // 4 * angular_out * angular_out, kernel_size=1,
        #               stride=1, padding=0, bias=False),
        #     nn.LeakyReLU(0.1, inplace=True),
        #     nn.Conv3d(channel // 4 * angular_out * angular_out, angular_out * angular_out, kernel_size=1, stride=1,
        #               padding=0, bias=False),
        # )


    def forward(self, x, angout=None):
        angout = self.angRes_out if angout is None else angout
        buffer = self.extract_lf_feature(x)
        buffer = self.epiFeatureRebuild(buffer, angout)
        buffer = rearrange(buffer, 'b c u v h w -> b c (u v) h w')
        b, c, n, h, w = buffer.shape

        # buffer = buffer.contiguous().view(b, c * n, 1, h, w)
        buffer = self.DownSample(buffer).view(b, 1, angout * angout, h, w)  # n == angRes * angRes
        out = FormOutput(buffer, angout)
        return self.copy_input_corners_to_sai(out, x, angout)

    def extract_lf_feature(self, x):
        return self.feature_extractor(x)

    def forward_view(self, x, angout, u_idx, v_idx):
        if self.is_corner_view(angout, u_idx, v_idx):
            return self.input_corner_view(x, u_idx, v_idx)
        buffer = self.extract_lf_feature(x)
        view_feature = self.epiFeatureRebuild.forward_view(buffer, angout, u_idx, v_idx)
        view_feature = view_feature.unsqueeze(2)
        return self.DownSample(view_feature)[:, :, 0]

    @staticmethod
    def is_corner_view(angout, u_idx, v_idx):
        return (u_idx, v_idx) in (
            (0, 0),
            (0, angout - 1),
            (angout - 1, 0),
            (angout - 1, angout - 1)
        )

    def input_corner_view(self, x, u_idx, v_idx):
        _, _, height, width = x.shape
        h = height // self.angRes
        w = width // self.angRes
        in_u = 0 if u_idx == 0 else self.angRes - 1
        in_v = 0 if v_idx == 0 else self.angRes - 1
        return x[:, :, in_u * h:(in_u + 1) * h, in_v * w:(in_v + 1) * w]

    def copy_input_corners_to_sai(self, out, x, angout):
        _, _, out_height, out_width = out.shape
        h = out_height // angout
        w = out_width // angout
        out = out.clone()
        out[:, :, :h, :w] = self.input_corner_view(x, 0, 0)
        out[:, :, :h, -w:] = self.input_corner_view(x, 0, angout - 1)
        out[:, :, -h:, :w] = self.input_corner_view(x, angout - 1, 0)
        out[:, :, -h:, -w:] = self.input_corner_view(x, angout - 1, angout - 1)
        return out

class ASRFeatureExtractor(nn.Module):
    def __init__(self, angRes, channels=64):
        super(ASRFeatureExtractor, self).__init__()
        self.angRes = angRes
        n_group = 5
        n_block = 5
        self.conv_init0 = nn.Sequential(
            nn.Conv3d(1, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False)
        )
        self.conv_init = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.disentg = CascadeDisentgGroup(n_group, n_block, self.angRes, channels)

    def forward(self, x):
        buffer = rearrange(x, 'b c (u h) (v w) -> b c (u v) h w', u=self.angRes, v=self.angRes)
        buffer = self.conv_init0(buffer)
        buffer = self.conv_init(buffer) + buffer
        buffer = rearrange(buffer, "b c (u v) h w -> b c (h u) (w v)", u=self.angRes, v=self.angRes)
        buffer = self.disentg(buffer)
        return rearrange(buffer, "b c (h u) (w v) -> b c u v h w", u=self.angRes, v=self.angRes)


class CascadeDisentgGroup(nn.Module):
    def __init__(self, n_group, n_block, angRes, channels):
        super(CascadeDisentgGroup, self).__init__()
        self.n_group = n_group
        Groups = []
        for i in range(n_group):
            Groups.append(DisentgGroup(n_block, angRes, channels))
        self.Group = nn.Sequential(*Groups)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes),
                              bias=False)

    def forward(self, x):
        buffer = x
        for i in range(self.n_group):
            buffer = self.Group[i](buffer)
        return self.conv(buffer) + x


class DisentgGroup(nn.Module):
    def __init__(self, n_block, angRes, channels):
        super(DisentgGroup, self).__init__()
        self.n_block = n_block
        Blocks = []
        for i in range(n_block):
            Blocks.append(DisentgBlock(angRes, channels))
        self.Block = nn.Sequential(*Blocks)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes),
                              bias=False)

    def forward(self, x):
        buffer = x
        for i in range(self.n_block):
            buffer = self.Block[i](buffer)
        return self.conv(buffer) + x


class DisentgBlock(nn.Module):
    def __init__(self, angRes, channels):
        super(DisentgBlock, self).__init__()

        # intra spatial feature
        self.IntraSpaConv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes),
                      bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes),
                      bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes),
                      bias=False)
        )

        # inter spatial feature
        self.InterSpaConv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=1, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=1, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=1, padding=1, bias=False)
        )

        # intra angular feature
        self.IntraAngConv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=int(angRes), stride=int(angRes), dilation=1,
                      padding=0, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, int(angRes * angRes * channels), kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2),
            nn.PixelShuffle(angRes),
        )

        # inter angular feature
        self.InterAngConv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=int(angRes * 2), stride=int(angRes * 2),
                      dilation=1, padding=0, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, int(4 * angRes * angRes * channels), kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2),
            nn.PixelShuffle(2 * angRes),
        )

        # horizontal EPI feature
        self.HEPIConv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=[1, 2 * angRes + 1], stride=[1, 1], padding=[0, angRes],
                      bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, kernel_size=[1, 2 * angRes + 1], stride=[1, 1], padding=[0, angRes],
                      bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, kernel_size=[1, 2 * angRes + 1], stride=[1, 1], padding=[0, angRes],
                      bias=False)
        )

        # vertical EPI feature
        self.VEPIConv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=[2 * angRes + 1, 1], stride=[1, 1], padding=[angRes, 0],
                      bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, kernel_size=[2 * angRes + 1, 1], stride=[1, 1], padding=[angRes, 0],
                      bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, kernel_size=[2 * angRes + 1, 1], stride=[1, 1], padding=[angRes, 0],
                      bias=False)
        )

    def forward(self, x):
        feaIntraSpa = self.IntraSpaConv(x)
        buffer = feaIntraSpa + x
        feaInterSpa = self.InterSpaConv(buffer)
        buffer = feaInterSpa + buffer

        feaIntraAng = self.IntraAngConv(buffer)
        buffer = feaIntraAng + buffer
        feaInterAng = self.InterAngConv(buffer)
        buffer = feaInterAng + buffer

        feaHEPI = self.HEPIConv(buffer)
        buffer = feaHEPI + buffer
        feaVEPI = self.VEPIConv(buffer)
        out = feaVEPI + buffer
        return out
class EpiFeatureRebuild(nn.Module):
    def __init__(self, angRes_in, angRes_out, channels, feat_unfold=True, local_ensemble=False, cell_decode=False):
        super().__init__()
        self.angRes_in = angRes_in
        self.angRes_out = angRes_out
        self.feat_unfold = feat_unfold
        self.local_ensemble = local_ensemble
        self.cell_decode = cell_decode

        imnet_in_dim = channels
        if self.feat_unfold:
            imnet_in_dim *= 9
        imnet_in_dim += 2  # attach coord
        if self.cell_decode:
            imnet_in_dim += 2
        self.imnet = MLP(in_dim=imnet_in_dim, out_dim=channels, hidden_list=[256, 256, 256, 256])

    def query_feature(self, Feature, coord, cell=None):
        feat = Feature

        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])

        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-2:], flatten=False).to(feat.device) \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([q_feat, rel_coord], dim=-1)

                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    inp = torch.cat([inp, rel_cell], dim=-1)

                bs, q = coord.shape[:2]
                pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0]
            areas[0] = areas[3]
            areas[3] = t

            t = areas[1]
            areas[1] = areas[2]
            areas[2] = t

        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        return ret

    def query_Epi(self, epi, angout):
        buh, c, v, w = epi.shape

        # 2 x W --> 7 x W
        coord = make_coord([angout, w]).to(epi.device) \
            .unsqueeze(0).expand(epi.shape[0], w * angout, 2)
        output_epi = self.query_feature(epi, coord, cell=None).permute(0, 2, 1) \
            .view(epi.shape[0], -1, angout, w)

        # buh, c, angRes_out, w
        return output_epi

    def forward(self, x, angout=None):
        angout = self.angRes_out if angout is None else angout
        batch_size, channle, u, v, h, w = x.shape

        # 2 x 2 x H x W --> 2 x 7 x H x W
        horizontal_x = self.query_Epi(rearrange(x, 'b c u v h w -> (b u h) c v w'), angout)
        x = rearrange(horizontal_x, '(b u h) c v w -> b c u v h w', b=batch_size, h=h)
        # 2 x 7 x H x W --> 7 x 7 x H x W
        vertical_x = self.query_Epi(rearrange(x, 'b c u v h w -> (b v w) c u h'), angout)
        output = rearrange(vertical_x, '(b v w) c u h -> b c u v h w', b=batch_size, w=w)

        return output


class PlaneResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def forward(self, x):
        return self.body(x) + x


class VolumeResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def forward(self, x):
        return self.body(x) + x


class PlaneAttentionPool(nn.Module):
    def __init__(self, channels, num_heads=4, dropout=0., ffn_ratio=2):
        super().__init__()
        self.query_token = nn.Parameter(torch.zeros(1, 1, channels))
        self.norm_q = nn.LayerNorm(channels)
        self.norm_kv = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
            bias=False
        )
        self.norm_ffn = nn.LayerNorm(channels)
        hidden_dim = channels * ffn_ratio
        self.feed_forward = nn.Sequential(
            nn.Linear(channels, hidden_dim, bias=False),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, channels, bias=False),
            nn.Dropout(dropout)
        )

    def forward(self, tokens):
        query = tokens.mean(dim=1, keepdim=True) + self.query_token
        pooled = self.attention(
            self.norm_q(query),
            self.norm_kv(tokens),
            tokens,
            need_weights=False
        )[0] + query
        pooled = self.feed_forward(self.norm_ffn(pooled)) + pooled
        return pooled[:, 0, :]


class FourPlaneAttentionProjector(nn.Module):
    def __init__(self, channels, num_heads=4, dropout=0.):
        super().__init__()
        self.st_pool = PlaneAttentionPool(channels, num_heads=num_heads, dropout=dropout)
        self.uv_pool = PlaneAttentionPool(channels, num_heads=num_heads, dropout=dropout)
        self.su_pool = PlaneAttentionPool(channels, num_heads=num_heads, dropout=dropout)
        self.tv_pool = PlaneAttentionPool(channels, num_heads=num_heads, dropout=dropout)

    def forward(self, st, uv, su, tv):
        b, c, u, v, h, w = st.shape

        st_tokens = st.permute(0, 4, 5, 2, 3, 1).reshape(b * h * w, u * v, c)
        st_plane = self.st_pool(st_tokens).view(b, h, w, c).permute(0, 3, 1, 2).contiguous()

        uv_tokens = uv.permute(0, 2, 3, 4, 5, 1).reshape(b * u * v, h * w, c)
        uv_plane = self.uv_pool(uv_tokens).view(b, u, v, c).permute(0, 3, 1, 2).contiguous()

        su_tokens = su.permute(0, 4, 2, 3, 5, 1).reshape(b * h * u, v * w, c)
        su_plane = self.su_pool(su_tokens).view(b, h, u, c).permute(0, 3, 1, 2).contiguous()

        tv_tokens = tv.permute(0, 5, 3, 2, 4, 1).reshape(b * w * v, u * h, c)
        tv_plane = self.tv_pool(tv_tokens).view(b, w, v, c).permute(0, 3, 1, 2).contiguous()

        return st_plane, uv_plane, su_plane, tv_plane


class FourPlaneCrossAttention(nn.Module):
    def __init__(self, channels, num_heads=4, dropout=0., ffn_ratio=2):
        super().__init__()
        self.norm_attn = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
            bias=False
        )
        self.norm_ffn = nn.LayerNorm(channels)
        hidden_dim = channels * ffn_ratio
        self.feed_forward = nn.Sequential(
            nn.Linear(channels, hidden_dim, bias=False),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, channels, bias=False),
            nn.Dropout(dropout)
        )

    def forward(self, st, uv, su, tv):
        b, c, u, v, h, w = st.shape
        tokens = torch.stack((st, uv, su, tv), dim=2)
        tokens = tokens.permute(0, 3, 4, 5, 6, 2, 1).reshape(b * u * v * h * w, 4, c)
        attn_input = self.norm_attn(tokens)
        tokens = self.attention(attn_input, attn_input, tokens, need_weights=False)[0] + tokens
        tokens = self.feed_forward(self.norm_ffn(tokens)) + tokens
        tokens = tokens.view(b, u, v, h, w, 4, c).permute(5, 0, 6, 1, 2, 3, 4).contiguous()
        return tokens[0], tokens[1], tokens[2], tokens[3]


class FourPlaneDecoderBase(nn.Module):
    def __init__(self, channels, feat_unfold=False, local_ensemble=False, cell_decode=False):
        super().__init__()
        self.feat_unfold = feat_unfold
        self.local_ensemble = local_ensemble
        self.cell_decode = cell_decode

    @staticmethod
    def missing_view_mask(angout, device):
        mask = torch.ones((angout, angout), dtype=torch.bool, device=device)
        mask[0, 0] = False
        mask[0, -1] = False
        mask[-1, 0] = False
        mask[-1, -1] = False
        return mask

    @staticmethod
    def build_query_coords(batch, height, width, angout, device):
        angular = make_coord([angout, angout]).to(device)
        spatial = make_coord([height, width]).to(device)
        a = angular.shape[0]
        p = spatial.shape[0]

        query_uv = angular[:, None, :].expand(a, p, 2).reshape(1, a * p, 2)
        query_st = spatial[None, :, :].expand(a, p, 2).reshape(1, a * p, 2)
        query_uv = query_uv.expand(batch, a * p, 2).contiguous()
        query_st = query_st.expand(batch, a * p, 2).contiguous()

        query_su = torch.stack((query_st[:, :, 0], query_uv[:, :, 0]), dim=-1)
        query_tv = torch.stack((query_st[:, :, 1], query_uv[:, :, 1]), dim=-1)

        return {
            'query_uv': query_uv,
            'st': query_st,
            'uv': query_uv,
            'su': query_su,
            'tv': query_tv
        }

    @staticmethod
    def build_view_query_coords(batch, height, width, angout, u_idx, v_idx, device):
        angular = make_coord([angout, angout], flatten=False).to(device)
        spatial = make_coord([height, width]).to(device)
        uv = angular[u_idx, v_idx].view(1, 1, 2).expand(batch, spatial.shape[0], 2).contiguous()
        st = spatial.unsqueeze(0).expand(batch, spatial.shape[0], 2).contiguous()
        su = torch.stack((st[:, :, 0], uv[:, :, 0]), dim=-1)
        tv = torch.stack((st[:, :, 1], uv[:, :, 1]), dim=-1)
        return {
            'query_uv': uv,
            'st': st,
            'uv': uv,
            'su': su,
            'tv': tv
        }

    def _sample_plane(self, plane, coord):
        if self.feat_unfold:
            plane = F.unfold(plane, 3, padding=1).view(
                plane.shape[0], plane.shape[1] * 9, plane.shape[2], plane.shape[3])
        sampled = F.grid_sample(
            plane,
            coord.flip(-1).unsqueeze(1),
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        )[:, :, 0, :]
        return sampled.permute(0, 2, 1)

    def _project_planes(self, st, uv, su, tv):
        return self.plane_projector(st, uv, su, tv)

    def _copy_input_corners(self, output, source):
        if source is None:
            return output
        output = output.clone()
        output[:, :, 0, 0] = source[:, :, 0, 0]
        output[:, :, 0, -1] = source[:, :, 0, -1]
        output[:, :, -1, 0] = source[:, :, -1, 0]
        output[:, :, -1, -1] = source[:, :, -1, -1]
        return output

class GaussianPlaneRasterizer(nn.Module):
    def __init__(self, channels, hidden_dim=64, kernel_size=5):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")
        self.kernel_size = kernel_size
        self.param_head = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_dim, 7, kernel_size=1, bias=True),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_dim, channels, kernel_size=1, bias=True),
        )

    @staticmethod
    def _coord_to_index(coord, size):
        return ((coord + 1.0) * size - 1.0) * 0.5

    def forward(self, plane, output_size):
        b, c, h, w = plane.shape
        out_h, out_w = int(output_size[0]), int(output_size[1])
        device, dtype = plane.device, plane.dtype

        params = self.param_head(plane)
        values = self.value_head(plane) + plane
        params = params.permute(0, 2, 3, 1).reshape(b, h * w, 7)
        values = values.reshape(b, c, h * w)

        anchors = make_coord([h, w]).to(device=device, dtype=dtype)
        center_y = self._coord_to_index(anchors[:, 0], out_h).view(1, h * w)
        center_x = self._coord_to_index(anchors[:, 1], out_w).view(1, h * w)

        scale_y = max(float(out_h) / float(h), 1.0)
        scale_x = max(float(out_w) / float(w), 1.0)
        sigma_y = F.softplus(params[..., 1]) * scale_y + 0.35
        sigma_x = F.softplus(params[..., 2]) * scale_x + 0.35
        rho = torch.tanh(params[..., 3]) * 0.95
        alpha = torch.sigmoid(params[..., 0]) * torch.sigmoid(params[..., 4])
        offset_y = torch.tanh(params[..., 5]) * 0.5 * scale_y
        offset_x = torch.tanh(params[..., 6]) * 0.5 * scale_x
        center_y = center_y + offset_y
        center_x = center_x + offset_x

        radius = self.kernel_size // 2
        offsets = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        dy, dx = torch.meshgrid(offsets, offsets)
        dy = dy.reshape(1, 1, -1)
        dx = dx.reshape(1, 1, -1)

        target_y = center_y.round().unsqueeze(-1) + dy
        target_x = center_x.round().unsqueeze(-1) + dx
        valid = (target_y >= 0) & (target_y < out_h) & (target_x >= 0) & (target_x < out_w)

        rel_y = target_y - center_y.unsqueeze(-1)
        rel_x = target_x - center_x.unsqueeze(-1)
        sigma_y = sigma_y.unsqueeze(-1)
        sigma_x = sigma_x.unsqueeze(-1)
        rho = rho.unsqueeze(-1)
        denom = (1.0 - rho ** 2).clamp_min(1e-4)
        exponent = (
            (rel_y / sigma_y) ** 2
            + (rel_x / sigma_x) ** 2
            - 2.0 * rho * rel_y * rel_x / (sigma_y * sigma_x)
        )
        weights = torch.exp(-0.5 * exponent / denom) * alpha.unsqueeze(-1)
        weights = weights * valid.to(dtype)

        safe_y = target_y.clamp(0, out_h - 1).long()
        safe_x = target_x.clamp(0, out_w - 1).long()
        flat_index = (safe_y * out_w + safe_x).reshape(b, -1)

        weights_flat = weights.reshape(b, -1)
        weighted_values = (
            values.unsqueeze(-1) * weights.unsqueeze(1)
        ).reshape(b, c, -1)

        field = plane.new_zeros(b, c, out_h * out_w)
        norm = plane.new_zeros(b, 1, out_h * out_w)
        field.scatter_add_(2, flat_index.unsqueeze(1).expand(-1, c, -1), weighted_values)
        norm.scatter_add_(2, flat_index.unsqueeze(1), weights_flat.unsqueeze(1))

        field = field / norm.clamp_min(1e-6)
        fallback = F.interpolate(values.view(b, c, h, w), size=(out_h, out_w), mode='bilinear', align_corners=False)
        fallback = fallback.view(b, c, out_h * out_w)
        field = torch.where(norm > 1e-6, field, fallback)
        return field.view(b, c, out_h, out_w)


class VolumeAttentionProjector(nn.Module):
    def __init__(self, channels, num_heads=4, dropout=0.):
        super().__init__()
        self.stu_pool = PlaneAttentionPool(channels, num_heads=num_heads, dropout=dropout)
        self.stv_pool = PlaneAttentionPool(channels, num_heads=num_heads, dropout=dropout)
        self.suv_pool = PlaneAttentionPool(channels, num_heads=num_heads, dropout=dropout)
        self.tuv_pool = PlaneAttentionPool(channels, num_heads=num_heads, dropout=dropout)

    def forward(self, stu, stv, suv, tuv):
        b, c, u, v, s, t = stu.shape

        stu_tokens = stu.permute(0, 4, 5, 2, 3, 1).reshape(b * s * t * u, v, c)
        stu_volume = self.stu_pool(stu_tokens).view(b, s, t, u, c).permute(0, 4, 1, 2, 3).contiguous()

        stv_tokens = stv.permute(0, 4, 5, 3, 2, 1).reshape(b * s * t * v, u, c)
        stv_volume = self.stv_pool(stv_tokens).view(b, s, t, v, c).permute(0, 4, 1, 2, 3).contiguous()

        suv_tokens = suv.permute(0, 4, 2, 3, 5, 1).reshape(b * s * u * v, t, c)
        suv_volume = self.suv_pool(suv_tokens).view(b, s, u, v, c).permute(0, 4, 1, 2, 3).contiguous()

        tuv_tokens = tuv.permute(0, 5, 2, 3, 4, 1).reshape(b * t * u * v, s, c)
        tuv_volume = self.tuv_pool(tuv_tokens).view(b, t, u, v, c).permute(0, 4, 1, 2, 3).contiguous()

        return stu_volume, stv_volume, suv_volume, tuv_volume


class GaussianVolumeRasterizer(nn.Module):
    def __init__(self, channels, hidden_dim=64, kernel_size=5, num_gaussians=4):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")
        if num_gaussians < 1:
            raise ValueError("num_gaussians must be >= 1.")
        self.kernel_size = kernel_size
        self.num_gaussians = num_gaussians
        self.param_head = nn.Sequential(
            nn.Conv3d(channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(hidden_dim, 10 * num_gaussians, kernel_size=1, bias=True),
        )
        self.value_head = nn.Sequential(
            nn.Conv3d(channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(hidden_dim, channels, kernel_size=1, bias=True),
        )

    @staticmethod
    def _coord_to_index(coord, size):
        return ((coord + 1.0) * size - 1.0) * 0.5

    def forward(self, volume, output_size):
        b, c, d, h, w = volume.shape
        out_d, out_h, out_w = [int(v) for v in output_size]
        device, dtype = volume.device, volume.dtype
        n = d * h * w
        g = self.num_gaussians

        params = self.param_head(volume)
        values = self.value_head(volume) + volume
        params = params.permute(0, 2, 3, 4, 1).reshape(b, n, g, 10)
        values = values.reshape(b, c, n)
        values = values.unsqueeze(3).expand(-1, -1, -1, g).reshape(b, c, n * g)

        anchors = make_coord([d, h, w]).to(device=device, dtype=dtype)
        center_d = self._coord_to_index(anchors[:, 0], out_d).view(1, n, 1)
        center_h = self._coord_to_index(anchors[:, 1], out_h).view(1, n, 1)
        center_w = self._coord_to_index(anchors[:, 2], out_w).view(1, n, 1)

        scale_d = max(float(out_d) / float(d), 1.0)
        scale_h = max(float(out_h) / float(h), 1.0)
        scale_w = max(float(out_w) / float(w), 1.0)
        alpha = torch.sigmoid(params[..., 0])
        offset_d = torch.tanh(params[..., 1]) * 0.5 * scale_d
        offset_h = torch.tanh(params[..., 2]) * 0.5 * scale_h
        offset_w = torch.tanh(params[..., 3]) * 0.5 * scale_w
        center_d = center_d + offset_d
        center_h = center_h + offset_h
        center_w = center_w + offset_w

        chol = params[..., 4:10]
        l00 = F.softplus(chol[..., 0]) * scale_d + 0.35
        l10 = torch.tanh(chol[..., 1]) * 0.5 * math.sqrt(scale_d * scale_h)
        l11 = F.softplus(chol[..., 2]) * scale_h + 0.35
        l20 = torch.tanh(chol[..., 3]) * 0.5 * math.sqrt(scale_d * scale_w)
        l21 = torch.tanh(chol[..., 4]) * 0.5 * math.sqrt(scale_h * scale_w)
        l22 = F.softplus(chol[..., 5]) * scale_w + 0.35

        cov = volume.new_zeros(b, n, g, 3, 3)
        cov[..., 0, 0] = l00 * l00
        cov[..., 0, 1] = l00 * l10
        cov[..., 1, 0] = cov[..., 0, 1]
        cov[..., 0, 2] = l00 * l20
        cov[..., 2, 0] = cov[..., 0, 2]
        cov[..., 1, 1] = l10 * l10 + l11 * l11
        cov[..., 1, 2] = l10 * l20 + l11 * l21
        cov[..., 2, 1] = cov[..., 1, 2]
        cov[..., 2, 2] = l20 * l20 + l21 * l21 + l22 * l22
        eye = torch.eye(3, device=device, dtype=dtype).view(1, 1, 1, 3, 3)
        inv_cov = torch.linalg.inv(cov + eye * 1e-4)

        radius = self.kernel_size // 2
        offsets = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        od, oh, ow = torch.meshgrid(offsets, offsets, offsets)
        od = od.reshape(1, 1, 1, -1)
        oh = oh.reshape(1, 1, 1, -1)
        ow = ow.reshape(1, 1, 1, -1)

        target_d = center_d.round().unsqueeze(-1) + od
        target_h = center_h.round().unsqueeze(-1) + oh
        target_w = center_w.round().unsqueeze(-1) + ow
        valid = (target_d >= 0) & (target_d < out_d) & (target_h >= 0) & (target_h < out_h) & (target_w >= 0) & (target_w < out_w)

        rel = torch.stack((
            target_d - center_d.unsqueeze(-1),
            target_h - center_h.unsqueeze(-1),
            target_w - center_w.unsqueeze(-1),
        ), dim=-1)
        exponent = torch.einsum('bngki,bngij,bngkj->bngk', rel, inv_cov, rel)
        weights = torch.exp(-0.5 * exponent.clamp_max(80.0)) * alpha.unsqueeze(-1)
        weights = weights * valid.to(dtype)

        safe_d = target_d.clamp(0, out_d - 1).long()
        safe_h = target_h.clamp(0, out_h - 1).long()
        safe_w = target_w.clamp(0, out_w - 1).long()
        flat_index = (safe_d * out_h * out_w + safe_h * out_w + safe_w).reshape(b, -1)

        weights_flat = weights.reshape(b, -1)
        weighted_values = (
            values.unsqueeze(-1) * weights.reshape(b, 1, n * g, -1)
        ).reshape(b, c, -1)

        field = volume.new_zeros(b, c, out_d * out_h * out_w)
        norm = volume.new_zeros(b, 1, out_d * out_h * out_w)
        field.scatter_add_(2, flat_index.unsqueeze(1).expand(-1, c, -1), weighted_values)
        norm.scatter_add_(2, flat_index.unsqueeze(1), weights_flat.unsqueeze(1))

        field = field / norm.clamp_min(1e-6)
        fallback = F.interpolate(volume, size=(out_d, out_h, out_w), mode='trilinear', align_corners=False)
        fallback = fallback.view(b, c, out_d * out_h * out_w)
        field = torch.where(norm > 1e-6, field, fallback)
        return field.view(b, c, out_d, out_h, out_w)

class VolumeAxialTransformerBlock(nn.Module):
    def __init__(self, channels, num_heads=4, dropout=0., ffn_ratio=2):
        super().__init__()
        self.axis_attn = nn.ModuleList([
            nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True, bias=False)
            for _ in range(3)
        ])
        self.axis_norm = nn.ModuleList([nn.LayerNorm(channels) for _ in range(3)])
        self.ffn_norm = nn.LayerNorm(channels)
        hidden_dim = channels * ffn_ratio
        self.ffn = nn.Sequential(
            nn.Linear(channels, hidden_dim, bias=False),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, channels, bias=False),
            nn.Dropout(dropout)
        )

    def _attend_axis(self, volume, axis, attn, norm):
        b, c, d0, d1, d2 = volume.shape
        tokens = volume.permute(0, 2, 3, 4, 1)
        if axis == 0:
            tokens = tokens.reshape(b, d0, d1 * d2, c).permute(0, 2, 1, 3).reshape(b * d1 * d2, d0, c)
            shape = (b, d1, d2, d0, c)
            inv = lambda x: x.view(*shape).permute(0, 3, 1, 2, 4)
        elif axis == 1:
            tokens = tokens.permute(0, 2, 4, 3, 1).reshape(b * d0 * d2, d1, c)
            shape = (b, d0, d2, d1, c)
            inv = lambda x: x.view(*shape).permute(0, 1, 3, 2, 4)
        else:
            tokens = tokens.reshape(b * d0 * d1, d2, c)
            shape = (b, d0, d1, d2, c)
            inv = lambda x: x.view(*shape)
        attn_input = norm(tokens)
        tokens = attn(attn_input, attn_input, tokens, need_weights=False)[0] + tokens
        return inv(tokens).permute(0, 4, 1, 2, 3).contiguous()

    def forward(self, volume):
        for axis, (attn, norm) in enumerate(zip(self.axis_attn, self.axis_norm)):
            volume = self._attend_axis(volume, axis, attn, norm)
        tokens = volume.permute(0, 2, 3, 4, 1)
        tokens = self.ffn(self.ffn_norm(tokens)) + tokens
        return tokens.permute(0, 4, 1, 2, 3).contiguous()


class FourVolumeSummaryTransformer(nn.Module):
    volume_names = ('uvs', 'uvt', 'ust', 'vst')

    def __init__(self, channels, num_heads=4, dropout=0., pooled_spatial_size=32):
        super().__init__()
        self.pooled_spatial_size = pooled_spatial_size
        self.uvs_axial = VolumeAxialTransformerBlock(channels, num_heads=num_heads, dropout=dropout)
        self.uvt_axial = VolumeAxialTransformerBlock(channels, num_heads=num_heads, dropout=dropout)
        self.ust_axial = VolumeAxialTransformerBlock(channels, num_heads=num_heads, dropout=dropout)
        self.vst_axial = VolumeAxialTransformerBlock(channels, num_heads=num_heads, dropout=dropout)
        self.cross_type_tokens = nn.Parameter(torch.zeros(1, 5, channels))
        self.cross_volume_norm = nn.LayerNorm(channels)
        self.cross_volume_attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
            bias=False
        )
        self.cross_ffn_norm = nn.LayerNorm(channels)
        self.cross_ffn = nn.Sequential(
            nn.Linear(channels, channels * 2, bias=False),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels, bias=False),
            nn.Dropout(dropout)
        )
        self.volume_delta = nn.ModuleList([nn.Linear(channels, channels, bias=False) for _ in range(4)])

    @staticmethod
    def _volume_full_tokens(volume):
        b, c, d0, d1, d2 = volume.shape
        tokens = volume.permute(0, 2, 3, 4, 1).reshape(b, d0 * d1 * d2, c)
        return tokens, (d0, d1, d2)

    @staticmethod
    def _tokens_to_full_volume(tokens, shape):
        b, _, c = tokens.shape
        d0, d1, d2 = shape
        return tokens.view(b, d0, d1, d2, c).permute(0, 4, 1, 2, 3).contiguous()

    def _volume_pooled_tokens(self, volume):
        b, c, d0, d1, d2 = volume.shape
        pooled_shape = (d0, min(self.pooled_spatial_size, d1), min(self.pooled_spatial_size, d2))
        pooled = F.adaptive_avg_pool3d(volume, pooled_shape)
        tokens = pooled.permute(0, 2, 3, 4, 1).reshape(b, pooled_shape[0] * pooled_shape[1] * pooled_shape[2], c)
        return tokens, (d0, d1, d2), pooled_shape

    @staticmethod
    def _tokens_to_pooled_volume(tokens, original_shape, pooled_shape):
        b, _, c = tokens.shape
        volume = tokens.view(b, pooled_shape[0], pooled_shape[1], pooled_shape[2], c).permute(0, 4, 1, 2, 3).contiguous()
        if tuple(pooled_shape) != tuple(original_shape):
            volume = F.interpolate(volume, size=original_shape, mode='trilinear', align_corners=False)
        return volume

    def _add_type_token(self, tokens, type_idx):
        return tokens + self.cross_type_tokens[:, type_idx:type_idx + 1]

    def forward(self, volumes, source, ablate_volumes=None):
        if isinstance(ablate_volumes, str):
            ablate_volumes = {ablate_volumes}
        else:
            ablate_volumes = set(ablate_volumes or ())
        uvs, uvt, ust, vst = volumes
        volume_items = {
            'uvs': uvs,
            'uvt': uvt,
            'ust': ust,
            'vst': vst,
        }
        active_volume_names = tuple(name for name in self.volume_names if name not in ablate_volumes)
        active_volumes = {}
        active_seeds = []
        active_meta = {}
        axial_blocks = {
            'uvs': self.uvs_axial,
            'uvt': self.uvt_axial,
            'ust': self.ust_axial,
            'vst': self.vst_axial,
        }
        for name in active_volume_names:
            volume = axial_blocks[name](volume_items[name])
            active_volumes[name] = volume
            if name in ('uvs', 'uvt'):
                volume_tokens, volume_shape = self._volume_full_tokens(volume)
                pooled_shape = None
            else:
                volume_tokens, volume_shape, pooled_shape = self._volume_pooled_tokens(volume)
            type_idx = self.volume_names.index(name)
            active_seeds.append(self._add_type_token(volume_tokens, type_idx))
            active_meta[name] = (volume_shape, pooled_shape)

        lf_token = source.mean(dim=(2, 3, 4, 5)).unsqueeze(1)
        lf_seed = self._add_type_token(lf_token, 4)
        tokens = torch.cat(active_seeds + [lf_seed], dim=1)
        lengths = [seed.shape[1] for seed in active_seeds] + [lf_seed.shape[1]]

        attn_input = self.cross_volume_norm(tokens)
        tokens = self.cross_volume_attn(attn_input, attn_input, tokens, need_weights=False)[0] + tokens
        tokens = self.cross_ffn(self.cross_ffn_norm(tokens)) + tokens
        active_cross = torch.split(tokens, lengths, dim=1)
        outputs = {
            name: torch.zeros_like(volume_items[name])
            for name in ablate_volumes
            if name in volume_items
        }
        for idx, name in enumerate(active_volume_names):
            volume_shape, pooled_shape = active_meta[name]
            delta = self.volume_delta[self.volume_names.index(name)](
                active_cross[idx] - active_seeds[idx]
            )
            if pooled_shape is None:
                outputs[name] = active_volumes[name] + self._tokens_to_full_volume(delta, volume_shape)
            else:
                outputs[name] = active_volumes[name] + self._tokens_to_pooled_volume(
                    delta, volume_shape, pooled_shape
                )
        return outputs['uvs'], outputs['uvt'], outputs['ust'], outputs['vst']

class VolumeFieldUpsampler(nn.Module):
    def __init__(self, channels, stride):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose3d(
                channels,
                channels,
                kernel_size=tuple(4 if s > 1 else 3 for s in stride),
                stride=stride,
                padding=tuple(1 for _ in stride),
                bias=False
            ),
            nn.LeakyReLU(0.1, inplace=True),
            VolumeResidualBlock(channels),
            VolumeResidualBlock(channels),
        )

    def forward(self, volume, output_size):
        field = self.up(volume)
        if tuple(field.shape[-3:]) != tuple(output_size):
            field = F.interpolate(field, size=output_size, mode='trilinear', align_corners=False)
        return field


class CoordinateConditionedBasisFieldUpsampler(nn.Module):
    """Synthesizes a requested angular field directly from a 2 / 2x2 source volume."""

    def __init__(self, channels, angular_dims, basis_count=8, groups=8, fourier_freqs=4):
        super().__init__()
        if angular_dims not in (1, 2):
            raise ValueError('angular_dims must be 1 or 2')
        if channels % groups != 0:
            raise ValueError('channels must be divisible by groups')
        if basis_count < 1:
            raise ValueError('basis_count must be positive')
        self.channels = channels
        self.angular_dims = angular_dims
        self.basis_count = basis_count
        self.groups = groups
        self.fourier_freqs = fourier_freqs
        self.basis_encoder = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(channels, channels * basis_count, kernel_size=1, bias=False),
        )
        self.content_encoder = nn.Sequential(
            nn.Linear(channels, channels, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
        )
        coordinate_dim = angular_dims * 2
        encoded_coordinate_dim = coordinate_dim * (1 + 2 * fourier_freqs)
        self.coefficient_decoder = nn.Sequential(
            nn.Linear(channels + encoded_coordinate_dim, channels, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(channels, groups * basis_count, bias=False),
        )

    def _encode_coordinates(self, coord):
        parts = [coord]
        if self.fourier_freqs:
            frequency = (2.0 ** torch.arange(
                self.fourier_freqs, device=coord.device, dtype=coord.dtype)) * math.pi
            phase = coord.unsqueeze(-1) * frequency
            parts.extend((torch.sin(phase).flatten(-2), torch.cos(phase).flatten(-2)))
        return torch.cat(parts, dim=-1)

    def _target_coordinates(self, angout, device, dtype):
        if angout <= 1:
            raise ValueError('angout must be greater than one')
        axis = torch.linspace(-1.0, 1.0, angout, device=device, dtype=dtype)
        if self.angular_dims == 1:
            coordinate = axis.unsqueeze(-1)
        else:
            u, v = torch.meshgrid(axis, axis, indexing='ij')
            coordinate = torch.stack((u, v), dim=-1).reshape(-1, 2)
        cell = coordinate.new_full(coordinate.shape, 2.0 / float(angout - 1))
        return torch.cat((coordinate, cell), dim=-1)

    def forward(self, volume, angout):
        if volume.ndim != 5:
            raise ValueError('volume must be [B,C,D,H,W]')
        if self.angular_dims == 1:
            if volume.shape[2] != 2:
                raise ValueError('one-axis volume must have angular size 2')
            basis = self.basis_encoder(volume).mean(dim=2)
        else:
            if tuple(volume.shape[2:4]) != (2, 2):
                raise ValueError('two-axis volume must have angular shape 2x2')
            basis = self.basis_encoder(volume).mean(dim=(2, 3))

        batch = volume.shape[0]
        spatial_shape = basis.shape[2:]
        spatial_size = int(np.prod(spatial_shape))
        channels_per_group = self.channels // self.groups
        basis = basis.reshape(batch, self.groups, channels_per_group, self.basis_count, spatial_size)
        content = self.content_encoder(volume.mean(dim=tuple(range(2, volume.ndim))))
        coordinate = self._encode_coordinates(self._target_coordinates(angout, volume.device, volume.dtype))
        coordinate = coordinate.unsqueeze(0).expand(batch, -1, -1)
        content = content.unsqueeze(1).expand(-1, coordinate.shape[1], -1)
        coefficient = self.coefficient_decoder(torch.cat((content, coordinate), dim=-1))
        coefficient = coefficient.view(batch, -1, self.groups, self.basis_count)
        field = torch.einsum('bngr,bgcrs->bngcs', coefficient, basis)
        field = field.permute(0, 2, 3, 1, 4).reshape(batch, self.channels, -1, *spatial_shape)
        if self.angular_dims == 1:
            return field
        return field.reshape(batch, self.channels, angout, angout, *spatial_shape)


class FourVolumeGaussianDecoder(FourPlaneDecoderBase):
    volume_names = ('uvs', 'uvt', 'ust', 'vst')
    geometry_dim = 11
    # 一个高斯只写它自己 sigma_support_sigmas-sigma 马氏轮廓以内的像素；枚举窗口的半宽取
    # "这批里最宽高斯"的轮廓外框 = ceil(sigma_support_sigmas * max(sigma_s, sigma_t))。
    #
    # 开销与偏移数 n=(2*radius+1)^2 成正比（时间与 autograd 显存都按 n 线性涨），所以 cap
    # 必须收着用：
    #   cap=2 -> 精确覆盖 sigma<=0.67，n<=25，成本与"固定 5x5"完全相同
    #   cap=3 -> 精确覆盖 sigma<=1.00，n<=49，成本 2.0 倍
    #   cap=4 -> 精确覆盖 sigma<=1.33，n<=81，成本 3.2 倍（就是这一版触发 OOM 的设置）
    # 实测 sigma 收敛在 ~0.5，所以 cap=2 在收敛区间是精确的；sigma 超过 cap/3 时窗口会切掉
    # 3-sigma 椭圆的一部分，由归一化重新配权（相当于等效核略窄），属于可控退化、且没有
    # 梯度死区（对比之下直接 clamp sigma 会让 sigma 的梯度归零）。
    sigma_support_sigmas = 3.0
    splat_radius_cap = 2
    # 单批中间张量 (b, C, q*G, chunk) 的目标上限，用来给 offset 分批定步长。取大一些可以
    # 减少批次（每次分批都是一轮 kernel 启动），只要不超过这里的字节数即可。
    splat_chunk_target_bytes = 256 * 1024 * 1024

    def __init__(
            self,
            channels,
            feat_unfold=False,
            local_ensemble=False,
            cell_decode=False,
            num_gaussians=4,
            angle_pe_freqs=4,
            max_offset_px=12.0,
            use_adaptive_offset=True,
            adaptive_offset_min_px=8.0,
            adaptive_offset_max_px=16.0,
            use_confidence_gate=True,
            confidence_gate_blend=1.0,
            confidence_temperature=0.7,
            field_basis_count=16,
            field_basis_groups=8,
            field_fourier_freqs=4):
        super().__init__(
            channels,
            feat_unfold=feat_unfold,
            local_ensemble=local_ensemble,
            cell_decode=cell_decode
        )
        if feat_unfold:
            raise ValueError("FourVolumeGaussianDecoder does not support feat_unfold.")
        if cell_decode:
            raise ValueError("FourVolumeGaussianDecoder does not support cell_decode.")
        if num_gaussians < 1:
            raise ValueError("num_gaussians must be >= 1.")
        self.channels = channels
        self.num_gaussians = num_gaussians
        self.max_offset_px = float(max_offset_px)
        self.use_adaptive_offset = bool(use_adaptive_offset)
        self.adaptive_offset_min_px = float(adaptive_offset_min_px)
        self.adaptive_offset_max_px = float(adaptive_offset_max_px)
        self.use_confidence_gate = bool(use_confidence_gate)
        self.confidence_gate_blend = float(confidence_gate_blend)
        self.confidence_temperature = float(confidence_temperature)
        self.interaction = FourVolumeSummaryTransformer(channels, num_heads=4, dropout=0.)
        self.uvs_field_upsampler = CoordinateConditionedBasisFieldUpsampler(
            channels, angular_dims=2, basis_count=field_basis_count,
            groups=field_basis_groups, fourier_freqs=field_fourier_freqs)
        self.uvt_field_upsampler = CoordinateConditionedBasisFieldUpsampler(
            channels, angular_dims=2, basis_count=field_basis_count,
            groups=field_basis_groups, fourier_freqs=field_fourier_freqs)
        self.ust_field_upsampler = CoordinateConditionedBasisFieldUpsampler(
            channels, angular_dims=1, basis_count=field_basis_count,
            groups=field_basis_groups, fourier_freqs=field_fourier_freqs)
        self.vst_field_upsampler = CoordinateConditionedBasisFieldUpsampler(
            channels, angular_dims=1, basis_count=field_basis_count,
            groups=field_basis_groups, fourier_freqs=field_fourier_freqs)
        self.imnet = MLP(
            in_dim=channels * 5 + self.geometry_dim,
            out_dim=num_gaussians * (channels + 6),
            hidden_list=[256, 256, 256, 256]
        )
        self.confidence_net = nn.Linear(channels * 5 + self.geometry_dim, num_gaussians)
        self._init_confidence_gate_neutral()
        self.ablate_planes = set()
        self.current_gaussian_stats = []
        self.enable_gaussian_stats = False
        self.current_confidence_stats = {}
        self.current_offset_limit_stats = {}
        self.register_buffer('last_volume_ratios', torch.full((4,), 0.25), persistent=False)
        self.register_buffer('last_anchor_count', torch.zeros((), dtype=torch.long), persistent=False)
        self.register_buffer('last_gaussian_count', torch.zeros((), dtype=torch.long), persistent=False)

    def _init_confidence_gate_neutral(self):
        if isinstance(self.confidence_net, nn.Linear):
            nn.init.zeros_(self.confidence_net.weight)
            if self.confidence_net.bias is not None:
                nn.init.zeros_(self.confidence_net.bias)
            return
        final_layer = self.confidence_net.layers[-1]
        if isinstance(final_layer, nn.Linear):
            nn.init.zeros_(final_layer.weight)
            if final_layer.bias is not None:
                nn.init.zeros_(final_layer.bias)

    def _offset_limit_from_geometry(self, geometry):
        if not self.use_adaptive_offset:
            return geometry.new_full(geometry.shape[:-1] + (1,), self.max_offset_px)
        angular_distance = geometry[..., 8:9]
        max_distance = math.sqrt(8.0)
        normalized_distance = torch.clamp(angular_distance / max_distance, 0.0, 1.0)
        return self.adaptive_offset_min_px + normalized_distance * (self.adaptive_offset_max_px - self.adaptive_offset_min_px)

    def _decode_confidence(self, inp, batch_size, input_views, pixels):
        if not self.use_confidence_gate:
            return None
        logits = self.confidence_net(inp.reshape(batch_size * input_views * pixels, -1))
        logits = logits.view(batch_size, input_views, pixels, self.num_gaussians, 1)
        temperature = max(self.confidence_temperature, 1e-6)
        confidence = torch.softmax(logits / temperature, dim=1)
        blend = min(1.0, max(0.0, self.confidence_gate_blend))
        if blend < 1.0:
            uniform = confidence.new_full((), 1.0 / float(max(input_views, 1)))
            confidence = (1.0 - blend) * uniform + blend * confidence
        return confidence.reshape(batch_size, input_views * pixels, self.num_gaussians, 1)

    def _is_volume_ablated(self, volume_name):
        if self.ablate_planes is None:
            return False
        if isinstance(self.ablate_planes, str):
            return volume_name == self.ablate_planes
        return volume_name in self.ablate_planes

    def _build_fields(self, features, angout, source):
        uvs, uvt, ust, vst = self.interaction(features, source, self.ablate_planes)
        volumes = {'uvs': uvs, 'uvt': uvt, 'ust': ust, 'vst': vst}
        upsamplers = {
            'uvs': self.uvs_field_upsampler,
            'uvt': self.uvt_field_upsampler,
            'ust': self.ust_field_upsampler,
            'vst': self.vst_field_upsampler,
        }
        fields = {}
        for name, volume in volumes.items():
            if self._is_volume_ablated(name):
                fields[name] = self._zero_ablated_field(volume, name, angout)
            else:
                fields[name] = upsamplers[name](volume, angout)
        return fields

    @staticmethod
    def _zero_ablated_field(volume, name, angout):
        b, c = volume.shape[:2]
        if name in ('uvs', 'uvt'):
            return volume.new_zeros((b, c, angout, angout, volume.shape[-1]))
        return volume.new_zeros((b, c, angout, volume.shape[-2], volume.shape[-1]))

    @staticmethod
    def _coord_to_index(coord, size):
        return ((coord + 1.0) * size - 1.0) * 0.5

    @staticmethod
    def _sample_uv_to_physical_uv(sample_uv, angout):
        if angout <= 1:
            raise ValueError("angout must be greater than 1 for physical angular coordinates.")
        return sample_uv * (float(angout) / float(angout - 1))

    @staticmethod
    def _input_anchor_physical_uv(u, v, device, dtype):
        if u == 2 and v == 2:
            return torch.tensor([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]],
                                device=device, dtype=dtype)
        sample_uv = make_coord([u, v]).to(device=device, dtype=dtype)
        return sample_uv * (float(max(u, v)) / float(max(max(u, v) - 1, 1)))

    def _build_anchor_inputs(self, source, target_uv, angout):
        b, c, u, v, height, width = source.shape
        device, dtype = source.device, source.dtype
        input_uv_base = self._input_anchor_physical_uv(u, v, device, dtype)
        spatial_base = make_coord([height, width]).to(device=device, dtype=dtype)
        input_views = input_uv_base.shape[0]
        pixels = spatial_base.shape[0]

        input_uv_physical = input_uv_base[:, None, :].expand(input_views, pixels, 2).reshape(1, input_views * pixels, 2)
        spatial = spatial_base[None, :, :].expand(input_views, pixels, 2).reshape(1, input_views * pixels, 2)
        input_uv_physical = input_uv_physical.expand(b, input_views * pixels, 2).contiguous()
        spatial = spatial.expand(b, input_views * pixels, 2).contiguous()
        target_uv = target_uv.view(1, 1, 2).expand(b, input_views * pixels, 2).to(device=device, dtype=dtype)
        target_uv_physical = self._sample_uv_to_physical_uv(target_uv, angout)

        delta = target_uv_physical - input_uv_physical
        distance = torch.sqrt((delta ** 2).sum(dim=-1, keepdim=True).clamp_min(1e-12))
        direction = delta / distance.clamp_min(1e-6)
        direction = torch.where(distance > 1e-6, direction, torch.zeros_like(direction))
        geometry = torch.cat((
            input_uv_physical,
            target_uv_physical,
            delta,
            delta.abs(),
            distance,
            direction
        ), dim=-1)
        anchor_feature = source.permute(0, 2, 3, 4, 5, 1).reshape(b, input_views * pixels, c)
        return target_uv, spatial, geometry, anchor_feature

    def _query_anchor_features(self, fields, source, target_uv, angout):
        _, spatial, geometry, anchor_feature = self._build_anchor_inputs(source, target_uv, angout)
        batch, _, input_u, input_v, height, width = source.shape
        input_views = input_u * input_v
        pixels = height * width
        target_index = self._coord_to_index(target_uv, angout).round().long().clamp(0, angout - 1)
        target_u, target_v = target_index[0], target_index[1]
        contexts = {
            'uvs': fields['uvs'][:, :, target_u, target_v, :].unsqueeze(-1).expand(-1, -1, -1, width),
            'uvt': fields['uvt'][:, :, target_u, target_v, :].unsqueeze(-2).expand(-1, -1, height, -1),
            'ust': fields['ust'][:, :, target_u, :, :],
            'vst': fields['vst'][:, :, target_v, :, :],
        }
        sampled = []
        for name in self.volume_names:
            context = contexts[name].permute(0, 2, 3, 1).reshape(batch, pixels, self.channels)
            if self._is_volume_ablated(name):
                context = torch.zeros_like(context)
            sampled.append(context.unsqueeze(1).expand(-1, input_views, -1, -1).reshape(
                batch, input_views * pixels, self.channels))
        self._update_volume_ratios(sampled)
        return sampled, anchor_feature, geometry, spatial

    def _update_volume_ratios(self, sampled):
        with torch.no_grad():
            first_linear = self.imnet.layers[0]
            contributions = []
            for idx, volume_feature in enumerate(sampled):
                start = idx * self.channels
                end = start + self.channels
                feature_scale = volume_feature.detach().abs().mean(dim=(0, 1))
                weight_scale = first_linear.weight[:, start:end].detach().abs().sum(dim=0)
                contributions.append((feature_scale * weight_scale).sum())
            contributions = torch.stack(contributions)
            total = contributions.sum()
            if total.item() > 1e-12:
                ratios = contributions / total
            else:
                ratios = torch.zeros_like(contributions)
            self.last_volume_ratios.copy_(ratios)

    def _decode_gaussians(self, fields, source, target_uv, angout):
        sampled, anchor_feature, geometry, spatial = self._query_anchor_features(fields, source, target_uv, angout)
        inp = torch.cat((*sampled, anchor_feature, geometry), dim=-1)
        b, q = inp.shape[:2]
        input_views = source.shape[2] * source.shape[3]
        pixels = q // input_views
        params = self.imnet(inp.reshape(b * q, -1)).view(b, q, -1)
        offset_limit_px = self._offset_limit_from_geometry(geometry)
        confidence = self._decode_confidence(inp, b, input_views, pixels)
        return params, spatial, offset_limit_px, confidence

    def _scalar_stat(self, value, reducer):
        if value.numel() == 0:
            return 0.0
        return float(reducer(value.detach().to(torch.float32)).item())

    def _record_gaussian_stats(
            self,
            view_u,
            view_v,
            delta_tanh,
            delta,
            scale,
            opacity,
            spatial,
            height,
            width,
            offset_limit_px=None,
            confidence=None):
        with torch.no_grad():
            delta_px_s = delta[..., 0].abs() * (float(height) * 0.5)
            delta_px_t = delta[..., 1].abs() * (float(width) * 0.5)
            delta_px = torch.sqrt(delta_px_s * delta_px_s + delta_px_t * delta_px_t)
            saturation_s = delta_tanh[..., 0].abs() >= 0.98
            saturation_t = delta_tanh[..., 1].abs() >= 0.98
            proposed_s = spatial[..., 0].unsqueeze(-1) + delta[..., 0]
            proposed_t = spatial[..., 1].unsqueeze(-1) + delta[..., 1]
            clamped = (proposed_s < -1.0) | (proposed_s > 1.0) | (proposed_t < -1.0) | (proposed_t > 1.0)

            flat_delta = delta_px.reshape(-1).to(torch.float32)
            if flat_delta.numel() == 0:
                delta_p95 = 0.0
            else:
                p95_index = max(1, int(math.ceil(0.95 * float(flat_delta.numel()))))
                delta_p95 = float(torch.kthvalue(flat_delta, p95_index).values.item())
            stats = {
                'view_u': int(view_u) if view_u is not None else -1,
                'view_v': int(view_v) if view_v is not None else -1,
                'count': int(delta_px.numel()),
                'delta_px_mean': self._scalar_stat(delta_px, torch.mean),
                'delta_px_p95': delta_p95,
                'delta_px_max': self._scalar_stat(delta_px, torch.max),
                'delta_s_px_mean': self._scalar_stat(delta_px_s, torch.mean),
                'delta_s_px_max': self._scalar_stat(delta_px_s, torch.max),
                'delta_t_px_mean': self._scalar_stat(delta_px_t, torch.mean),
                'delta_t_px_max': self._scalar_stat(delta_px_t, torch.max),
                'sigma_s_mean': self._scalar_stat(scale[..., 0], torch.mean),
                'sigma_s_max': self._scalar_stat(scale[..., 0], torch.max),
                'sigma_t_mean': self._scalar_stat(scale[..., 1], torch.mean),
                'sigma_t_max': self._scalar_stat(scale[..., 1], torch.max),
                'opacity_mean': self._scalar_stat(opacity, torch.mean),
                'opacity_max': self._scalar_stat(opacity, torch.max),
                'tanh_sat_ratio': self._scalar_stat((saturation_s | saturation_t).to(torch.float32), torch.mean),
                'tanh_sat_s_ratio': self._scalar_stat(saturation_s.to(torch.float32), torch.mean),
                'tanh_sat_t_ratio': self._scalar_stat(saturation_t.to(torch.float32), torch.mean),
                'center_clamp_ratio': self._scalar_stat(clamped.to(torch.float32), torch.mean),
            }
            if offset_limit_px is not None:
                stats.update({
                    'offset_limit_px_mean': self._scalar_stat(offset_limit_px, torch.mean),
                    'offset_limit_px_max': self._scalar_stat(offset_limit_px, torch.max),
                })
            if confidence is not None and confidence.numel() > 0:
                conf = confidence.detach().to(torch.float32)
                stats['confidence_mean'] = self._scalar_stat(conf, torch.mean)
                if conf.shape[1] % 4 == 0:
                    pixels = conf.shape[1] // 4
                    conf_by_anchor = conf.reshape(conf.shape[0], 4, pixels, self.num_gaussians, 1)
                    conf_sum = conf_by_anchor.sum(dim=1).clamp_min(1e-8)
                    conf_norm = conf_by_anchor / conf_sum.unsqueeze(1)
                    conf_max = conf_norm.max(dim=1).values
                    entropy = -(conf_norm * conf_norm.clamp_min(1e-8).log()).sum(dim=1) / math.log(4.0)
                    stats.update({
                        'confidence_max_mean': self._scalar_stat(conf_max, torch.mean),
                        'confidence_entropy': self._scalar_stat(entropy, torch.mean),
                        'confidence_anchor_00_mean': self._scalar_stat(conf_by_anchor[:, 0], torch.mean),
                        'confidence_anchor_01_mean': self._scalar_stat(conf_by_anchor[:, 1], torch.mean),
                        'confidence_anchor_10_mean': self._scalar_stat(conf_by_anchor[:, 2], torch.mean),
                        'confidence_anchor_11_mean': self._scalar_stat(conf_by_anchor[:, 3], torch.mean),
                    })
            self.current_gaussian_stats.append(stats)

    def _splat_radius(self, sigma_s, sigma_t):
        """枚举窗口的半宽（像素）：恰好包住这批里最宽高斯的 sigma_support_sigmas 倍马氏椭圆。

        3-sigma 支撑域是 {x^T Sigma^-1 x <= 9}，即 x = 3*Sigma^{1/2}*u (|u|<=1)。它在 s 轴上的
        最大伸长是 3*sqrt(Sigma_ss) = 3*sigma_s，t 轴同理，**与 rho 无关**。所以轴对齐窗口
        取 ceil(support * max(sigma_s, sigma_t)) 就是精确外框。

        （早先按"Sigma 最大特征值开方"（半长轴）取半径是错的：那是椭圆沿对角方向的伸长，
        轴对齐窗口不需要它，白白把窗口放大到最多 1.396 倍。支撑域之外的点本来就被马氏门限
        置零，所以只要不触及 splat_radius_cap，改小窗口不改变任何输出，只省显存/算力；
        实测 sigma <= cap/3 时新旧规则输出逐位一致。）
        结果夹到 [1, splat_radius_cap]；超过上限时窗口会切掉 3-sigma 椭圆的一部分，截断后
        由归一化重新配权，属于可控退化（实测 sigma=1.6 时最坏约 0.5% 相对量级）。
        """
        with torch.no_grad():
            widest = max(float(sigma_s.detach().max().item()), float(sigma_t.detach().max().item()))
            needed = self.sigma_support_sigmas * widest
        if not math.isfinite(needed):
            return int(self.splat_radius_cap)
        return max(1, min(int(math.ceil(needed)), int(self.splat_radius_cap)))

    def _splat_gaussians(
            self,
            params,
            spatial,
            height,
            width,
            offset_limit_px=None,
            confidence=None,
            view_u=None,
            view_v=None):
        b, q, _ = params.shape
        channels = self.channels
        dtype, device = params.dtype, params.device
        params = params.view(b, q, self.num_gaussians, channels + 6)
        pixel_scale = 2.0 / float(max(min(height, width), 1))
        delta_tanh = torch.tanh(params[..., 0:2])
        if offset_limit_px is None:
            offset_limit_px = params.new_full((b, q, 1), self.max_offset_px)
        # Legacy multiplicative form: the limit is the reach cap and every splat keeps a
        # non-zero gradient through ``delta_tanh``, so no gaussian is ever frozen.
        if offset_limit_px.ndim == 3:
            offset_limit_px = offset_limit_px.unsqueeze(2)
        delta = delta_tanh * (offset_limit_px * pixel_scale)
        # sigma = exp(raw)，不再用手写地板。softplus 在 raw<0 时导数退化成 sigmoid(raw)
        # （实测 raw≈-4 处只有 0.018），sigma 被冻在 0.5 + 0.02 上动不了；exp 的相对导数
        # 恒为 1，sigma 才能真正被学到。clamp(-20, 20) 是纯数值保护而非取值边界：
        # 下端防 exp 下溢成精确 0 后 (rel/0) 出现 0/0 = NaN；上端防 exp 溢出成 inf 让
        # _splat_radius 的 ceil() 抛 OverflowError。sigma 实际范围是 [2.1e-9, 4.9e8]。
        # 注意：改了这里，所有用 softplus 版本训练的 checkpoint 都不能再直接评测
        # （它们的 raw≈-4 在新式下给出 sigma≈0.018，splat 核等于被压没），必须重新训练。
        scale = torch.exp(params[..., 2:4].clamp(-20.0, 20.0))
        rho = torch.tanh(params[..., 4:5]) * 0.95
        opacity = torch.sigmoid(params[..., 5:6])
        values = params[..., 6:6 + channels]
        if self.enable_gaussian_stats:
            self._record_gaussian_stats(
                view_u,
                view_v,
                delta_tanh,
                delta,
                scale,
                opacity,
                spatial,
                height,
                width,
                offset_limit_px=offset_limit_px,
                confidence=confidence
            )

        center_s = self._coord_to_index((spatial[..., 0].unsqueeze(-1) + delta[..., 0]).clamp(-1.0, 1.0), height)
        center_t = self._coord_to_index((spatial[..., 1].unsqueeze(-1) + delta[..., 1]).clamp(-1.0, 1.0), width)
        sigma_s = scale[..., 0:1]
        sigma_t = scale[..., 1:2]
        denom = (1.0 - rho ** 2).clamp_min(1e-4)

        radius = self._splat_radius(sigma_s, sigma_t)
        base_s = center_s.round()
        base_t = center_t.round()
        grid = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        ds_all = grid.repeat_interleave(grid.numel())
        dt_all = grid.repeat(grid.numel())

        values = values.permute(0, 3, 1, 2).reshape(b, channels, q, self.num_gaussians)
        field = params.new_zeros(b, channels, height * width)
        norm = params.new_zeros(b, 1, height * width)

        # 分批累加：同一批次内不物化 (b, C, q*G, n_offsets) 大张量，显存与偏移数无关，
        # 因此窗口可以跟随 sigma 放大。数学上与一次性物化完全等价。
        support_limit = self.sigma_support_sigmas ** 2
        offsets_total = (2 * radius + 1) ** 2
        # 步长按实际规模算，保证单批中间张量 (b, C, q*G, chunk) 不超过设定的字节上限。
        elements_per_offset = max(4 * b * channels * q * self.num_gaussians, 1)
        chunk = max(1, int(self.splat_chunk_target_bytes // elements_per_offset))
        for start in range(0, offsets_total, chunk):
            stop = min(start + chunk, offsets_total)
            ds = ds_all[start:stop].reshape(1, 1, 1, -1)
            dt = dt_all[start:stop].reshape(1, 1, 1, -1)

            target_s = base_s.unsqueeze(-1) + ds
            target_t = base_t.unsqueeze(-1) + dt
            rel_s = target_s - center_s.unsqueeze(-1)
            rel_t = target_t - center_t.unsqueeze(-1)
            # (rel^T Sigma^-1 rel) 正是高斯指数里的项，3-sigma 轮廓就是马氏距离 <= 9。
            mahalanobis = (
                (rel_s / sigma_s) ** 2
                + (rel_t / sigma_t) ** 2
                - 2.0 * rho * rel_s * rel_t / (sigma_s * sigma_t)
            ) / denom
            weights = torch.exp(-0.5 * mahalanobis) * opacity
            if confidence is not None:
                weights = weights * confidence
            # 3-sigma 支撑域只截掉权重 <= exp(-4.5) = 0.011 的远尾。实测（4 anchor × 4 高斯
            # 每像素）保留权重始终 >= 98%，且 sigma 从 0.05 扫到 2.5 都不会出现"某像素完全
            # 没被写入"——每个输出像素有多个 anchor 的高斯兜底。若将来 anchor 数或每像素
            # 高斯数被削减，需要复核这一条。
            valid = (
                (mahalanobis <= support_limit)
                & (target_s >= 0) & (target_s < height)
                & (target_t >= 0) & (target_t < width)
            )
            weights = weights * valid.to(dtype)

            safe_s = target_s.clamp(0, height - 1).long()
            safe_t = target_t.clamp(0, width - 1).long()
            flat_index = (safe_s * width + safe_t).reshape(b, -1)
            weights_flat = weights.reshape(b, -1)
            # values: (b, C, q, G)，weights: (b, q, G, n) -> 广播成 (b, C, q, G, n) 再展平，
            # 展平顺序必须与 flat_index 的 (q, G, n) 一致。
            weighted_values = (
                values.unsqueeze(-1)
                * weights.reshape(b, 1, q, self.num_gaussians, -1)
            ).reshape(b, channels, -1)
            field.scatter_add_(
                2,
                flat_index.unsqueeze(1).expand(-1, channels, -1),
                weighted_values
            )
            norm.scatter_add_(2, flat_index.unsqueeze(1), weights_flat.unsqueeze(1))

        field = field / norm.clamp_min(1e-6)
        return field.view(b, channels, height, width)
    def forward(self, uvs, uvt, ust, vst, angout, source=None):
        if source is None:
            raise ValueError("source LF features are required for input-view anchored Gaussian splatting.")
        b, _, _, _, height = uvs.shape
        _, _, _, _, width = uvt.shape
        self.current_gaussian_stats = []
        fields = self._build_fields((uvs, uvt, ust, vst), angout, source)
        angular = make_coord([angout, angout]).to(device=uvs.device, dtype=uvs.dtype)
        corner_indices = {
            (0, 0),
            (0, angout - 1),
            (angout - 1, 0),
            (angout - 1, angout - 1)
        }
        output_views = []
        for view_idx, target_uv in enumerate(angular):
            out_u = view_idx // angout
            out_v = view_idx % angout
            if (out_u, out_v) in corner_indices:
                input_u = 0 if out_u == 0 else source.shape[2] - 1
                input_v = 0 if out_v == 0 else source.shape[3] - 1
                output_views.append(source[:, :, input_u, input_v])
                continue
            params, spatial, offset_limit_px, confidence = self._decode_gaussians(
                fields, source, target_uv, angout)
            self.last_anchor_count.copy_(torch.tensor(params.shape[1], device=uvs.device, dtype=torch.long))
            self.last_gaussian_count.copy_(torch.tensor(params.shape[1] * self.num_gaussians, device=uvs.device, dtype=torch.long))
            output_views.append(self._splat_gaussians(
                params,
                spatial,
                height,
                width,
                offset_limit_px=offset_limit_px,
                confidence=confidence,
                view_u=out_u,
                view_v=out_v
            ))
        output = torch.stack(output_views, dim=2).view(b, self.channels, angout, angout, height, width)
        return self._copy_input_corners(output, source).contiguous()

    def forward_view(self, uvs, uvt, ust, vst, angout, u_idx, v_idx, source=None):
        if source is None:
            raise ValueError("source LF features are required for input-view anchored Gaussian splatting.")
        self.current_gaussian_stats = []
        if (u_idx, v_idx) in (
                (0, 0),
                (0, angout - 1),
                (angout - 1, 0),
                (angout - 1, angout - 1)):
            input_u = 0 if u_idx == 0 else source.shape[2] - 1
            input_v = 0 if v_idx == 0 else source.shape[3] - 1
            return source[:, :, input_u, input_v].contiguous()
        _, _, _, _, height = uvs.shape
        _, _, _, _, width = uvt.shape
        fields = self._build_fields((uvs, uvt, ust, vst), angout, source)
        angular = make_coord([angout, angout], flatten=False).to(device=uvs.device, dtype=uvs.dtype)
        params, spatial, offset_limit_px, confidence = self._decode_gaussians(
            fields, source, angular[u_idx, v_idx], angout)
        self.last_anchor_count.copy_(torch.tensor(params.shape[1], device=uvs.device, dtype=torch.long))
        self.last_gaussian_count.copy_(torch.tensor(params.shape[1] * self.num_gaussians, device=uvs.device, dtype=torch.long))
        return self._splat_gaussians(
            params,
            spatial,
            height,
            width,
            offset_limit_px=offset_limit_px,
            confidence=confidence,
            view_u=u_idx,
            view_v=v_idx
        ).contiguous()
class FourPlaneGaussianDecoder(FourPlaneDecoderBase):
    def __init__(
            self,
            channels,
            feat_unfold=False,
            local_ensemble=False,
            cell_decode=False,
            num_gaussians=4,
            angle_pe_freqs=4):
        super().__init__(
            channels,
            feat_unfold=feat_unfold,
            local_ensemble=local_ensemble,
            cell_decode=cell_decode
        )
        if cell_decode:
            raise ValueError("FourPlaneGaussianDecoder does not support cell_decode.")
        self.plane_projector = FourPlaneAttentionProjector(channels, num_heads=4, dropout=0.)
        sampled_dim = channels * 9 if self.feat_unfold else channels
        self.imnet = MLP(in_dim=sampled_dim * 4 + 4, out_dim=channels, hidden_list=[256, 256, 256, 256])
        self.st_rasterizer = GaussianPlaneRasterizer(channels, hidden_dim=64, kernel_size=5)
        self.uv_rasterizer = GaussianPlaneRasterizer(channels, hidden_dim=64, kernel_size=5)
        self.su_rasterizer = GaussianPlaneRasterizer(channels, hidden_dim=64, kernel_size=5)
        self.tv_rasterizer = GaussianPlaneRasterizer(channels, hidden_dim=64, kernel_size=5)
        self.ablate_planes = set()

    def _is_plane_ablated(self, plane_name):
        if self.ablate_planes is None:
            return False
        if isinstance(self.ablate_planes, str):
            return plane_name == self.ablate_planes
        return plane_name in self.ablate_planes

    def _build_fields(self, features, angout):
        st_plane, uv_plane, su_plane, tv_plane = self._project_planes(*features)
        _, _, h, w = st_plane.shape
        return {
            'st': self.st_rasterizer(st_plane, (h, w)),
            'uv': self.uv_rasterizer(uv_plane, (angout, angout)),
            'su': self.su_rasterizer(su_plane, (h, angout)),
            'tv': self.tv_rasterizer(tv_plane, (w, angout)),
        }

    @staticmethod
    def _coords_dict_to_uvst(coords):
        return torch.cat((coords['query_uv'], coords['st']), dim=-1)

    def _sample_named_plane(self, fields, plane_name, coord):
        sampled = self._sample_plane(fields[plane_name], coord)
        if self._is_plane_ablated(plane_name):
            sampled = torch.zeros_like(sampled)
        return sampled

    def query_points(self, features, coords, angout, fields=None):
        if coords.shape[-1] != 4:
            raise ValueError("coords must have shape [B, Q, 4] in (u, v, s, t) order.")
        fields = self._build_fields(features, angout) if fields is None else fields
        u, v, s, t = coords.unbind(dim=-1)
        plane_coords = {
            'uv': torch.stack((u, v), dim=-1),
            'st': torch.stack((s, t), dim=-1),
            'su': torch.stack((s, u), dim=-1),
            'tv': torch.stack((t, v), dim=-1),
        }

        q_st = self._sample_named_plane(fields, 'st', plane_coords['st'])
        q_uv = self._sample_named_plane(fields, 'uv', plane_coords['uv'])
        q_su = self._sample_named_plane(fields, 'su', plane_coords['su'])
        q_tv = self._sample_named_plane(fields, 'tv', plane_coords['tv'])
        inp = torch.cat((q_st, q_uv, q_su, q_tv, coords), dim=-1)
        bs, q = coords.shape[:2]
        return self.imnet(inp.reshape(bs * q, -1)).view(bs, q, -1)

    def query_feature(self, features, coords, angout=None, cell=None):
        if angout is None:
            b, _, _, _, h, w = features[0].shape
            q = coords['query_uv'].shape[1]
            views = max(q // (h * w), 1)
            angout = int(round(math.sqrt(views)))
        return self.query_points(features, self._coords_dict_to_uvst(coords), angout)

    def forward(self, st, uv, su, tv, angout, source=None):
        b, c, _, _, h, w = st.shape
        coords = self.build_query_coords(b, h, w, angout, st.device)
        coords_uvst = self._coords_dict_to_uvst(coords).to(dtype=st.dtype)
        fields = self._build_fields((st, uv, su, tv), angout)
        output = self.query_points((st, uv, su, tv), coords_uvst, angout, fields=fields)
        output = output.permute(0, 2, 1).view(b, c, angout, angout, h, w)
        return self._copy_input_corners(output, source).contiguous()

    def forward_view(self, st, uv, su, tv, angout, u_idx, v_idx):
        b, c, _, _, h, w = st.shape
        coords = self.build_view_query_coords(b, h, w, angout, u_idx, v_idx, st.device)
        coords_uvst = self._coords_dict_to_uvst(coords).to(dtype=st.dtype)
        output = self.query_points((st, uv, su, tv), coords_uvst, angout)
        return output.permute(0, 2, 1).view(b, c, h, w).contiguous()

class FourVolumeFeatureRebuild(nn.Module):
    def __init__(
            self,
            angRes_in,
            angRes_out,
            channels,
            feat_unfold=True,
            local_ensemble=False,
            cell_decode=False,
            decoder_type='gs',
            num_gaussians=4,
            use_volume_template=True,
            volume_template_size=64,
            volume_template_init_scale=0.1,
            max_offset_px=12.0,
            use_adaptive_offset=True,
            adaptive_offset_min_px=8.0,
            adaptive_offset_max_px=16.0,
            use_confidence_gate=True,
            confidence_gate_blend=1.0,
            confidence_temperature=0.7):
        super().__init__()
        self.angRes_in = angRes_in
        self.angRes_out = angRes_out
        self.channels = channels
        self.use_volume_template = use_volume_template
        self.volume_template_size = volume_template_size
        self.volume_template_std = float(volume_template_init_scale) / math.sqrt(channels)
        self.uvs_template = self._init_volume_template((1, channels, angRes_in, angRes_in, volume_template_size))
        self.uvt_template = self._init_volume_template((1, channels, angRes_in, angRes_in, volume_template_size))
        self.ust_template = self._init_volume_template((1, channels, angRes_in, volume_template_size, volume_template_size))
        self.vst_template = self._init_volume_template((1, channels, angRes_in, volume_template_size, volume_template_size))
        self.coord_proj = nn.Sequential(
            nn.Linear(4, channels, bias=False),
            nn.ReLU(True),
            nn.Linear(channels, channels, bias=False)
        )
        self.uvs_pool = PlaneAttentionPool(channels, num_heads=4, dropout=0.)
        self.uvt_pool = PlaneAttentionPool(channels, num_heads=4, dropout=0.)
        self.ust_pool = PlaneAttentionPool(channels, num_heads=4, dropout=0.)
        self.vst_pool = PlaneAttentionPool(channels, num_heads=4, dropout=0.)
        self.uvs_encoder = VolumeResidualBlock(channels)
        self.uvt_encoder = VolumeResidualBlock(channels)
        self.ust_encoder = VolumeResidualBlock(channels)
        self.vst_encoder = VolumeResidualBlock(channels)
        if decoder_type != 'gs':
            raise ValueError("decoder_type='mlp' has been removed; only decoder_type='gs' is supported.")
        self.implicit_decoder = FourVolumeGaussianDecoder(
            channels,
            feat_unfold=feat_unfold,
            local_ensemble=local_ensemble,
            cell_decode=cell_decode,
            num_gaussians=num_gaussians,
            max_offset_px=max_offset_px,
            use_adaptive_offset=use_adaptive_offset,
            adaptive_offset_min_px=adaptive_offset_min_px,
            adaptive_offset_max_px=adaptive_offset_max_px,
            use_confidence_gate=use_confidence_gate,
            confidence_gate_blend=confidence_gate_blend,
            confidence_temperature=confidence_temperature
        )

    def _init_volume_template(self, shape):
        return nn.Parameter(torch.randn(shape, dtype=torch.float32) * self.volume_template_std)

    def _add_template(self, volume, template, name):
        if not self.use_volume_template:
            return volume
        if volume.shape[1:] != template.shape[1:]:
            raise ValueError(
                "{} template shape {} does not match volume shape {}. "
                "This fixed-template experiment expects angin={} and spatial size {}.".format(
                    name, tuple(template.shape[1:]), tuple(volume.shape[1:]),
                    self.angRes_in, self.volume_template_size
                )
            )
        return volume + template.to(device=volume.device, dtype=volume.dtype)

    def apply_volume_templates(self, uvs, uvt, ust, vst):
        return (
            self._add_template(uvs, self.uvs_template, 'uvs'),
            self._add_template(uvt, self.uvt_template, 'uvt'),
            self._add_template(ust, self.ust_template, 'ust'),
            self._add_template(vst, self.vst_template, 'vst')
        )

    def _zero_ablated_volumes(self, uvs, uvt, ust, vst):
        """Zero template-conditioned branches before decoder interaction."""
        volumes = {'uvs': uvs, 'uvt': uvt, 'ust': ust, 'vst': vst}
        ablated = getattr(self.implicit_decoder, 'ablate_planes', set())
        if isinstance(ablated, str):
            ablated = {ablated}
        return tuple(
            torch.zeros_like(volumes[name]) if name in ablated else volumes[name]
            for name in ('uvs', 'uvt', 'ust', 'vst')
        )

    def add_coordinate_conditioning(self, x):
        b, c, u, v, s, t = x.shape
        coords = make_coord([u, v, s, t]).to(device=x.device, dtype=x.dtype)
        pos = self.coord_proj(coords).view(1, u, v, s, t, c).permute(0, 5, 1, 2, 3, 4)
        return x + pos

    def encode_uvs(self, x):
        b, c, u, v, s, t = x.shape
        tokens = x.permute(0, 2, 3, 4, 5, 1).reshape(b * u * v * s, t, c)
        volume = self.uvs_pool(tokens).view(b, u, v, s, c).permute(0, 4, 1, 2, 3).contiguous()
        return self.uvs_encoder(volume)

    def encode_uvt(self, x):
        b, c, u, v, s, t = x.shape
        tokens = x.permute(0, 2, 3, 5, 4, 1).reshape(b * u * v * t, s, c)
        volume = self.uvt_pool(tokens).view(b, u, v, t, c).permute(0, 4, 1, 2, 3).contiguous()
        return self.uvt_encoder(volume)

    def encode_ust(self, x):
        b, c, u, v, s, t = x.shape
        tokens = x.permute(0, 2, 4, 5, 3, 1).reshape(b * u * s * t, v, c)
        volume = self.ust_pool(tokens).view(b, u, s, t, c).permute(0, 4, 1, 2, 3).contiguous()
        return self.ust_encoder(volume)

    def encode_vst(self, x):
        b, c, u, v, s, t = x.shape
        tokens = x.permute(0, 3, 4, 5, 2, 1).reshape(b * v * s * t, u, c)
        volume = self.vst_pool(tokens).view(b, v, s, t, c).permute(0, 4, 1, 2, 3).contiguous()
        return self.vst_encoder(volume)

    def forward(self, x, angout=None):
        angout = self.angRes_out if angout is None else angout
        conditioned = self.add_coordinate_conditioning(x)
        uvs = self.encode_uvs(conditioned)
        uvt = self.encode_uvt(conditioned)
        ust = self.encode_ust(conditioned)
        vst = self.encode_vst(conditioned)
        uvs, uvt, ust, vst = self.apply_volume_templates(uvs, uvt, ust, vst)
        uvs, uvt, ust, vst = self._zero_ablated_volumes(uvs, uvt, ust, vst)
        return self.implicit_decoder(uvs, uvt, ust, vst, angout, source=x)

    def forward_view(self, x, angout, u_idx, v_idx):
        conditioned = self.add_coordinate_conditioning(x)
        uvs = self.encode_uvs(conditioned)
        uvt = self.encode_uvt(conditioned)
        ust = self.encode_ust(conditioned)
        vst = self.encode_vst(conditioned)
        uvs, uvt, ust, vst = self.apply_volume_templates(uvs, uvt, ust, vst)
        uvs, uvt, ust, vst = self._zero_ablated_volumes(uvs, uvt, ust, vst)
        return self.implicit_decoder.forward_view(uvs, uvt, ust, vst, angout, u_idx, v_idx, source=x)
class FourPlaneFeatureRebuild(nn.Module):
    def __init__(
            self,
            angRes_in,
            angRes_out,
            channels,
            feat_unfold=True,
            local_ensemble=False,
            cell_decode=False,
            decoder_type='gs',
            num_gaussians=4,
            max_offset_px=12.0,
            use_adaptive_offset=True,
            adaptive_offset_min_px=8.0,
            adaptive_offset_max_px=16.0,
            use_confidence_gate=True,
            confidence_gate_blend=1.0,
            confidence_temperature=0.7):
        super().__init__()
        self.angRes_in = angRes_in
        self.angRes_out = angRes_out
        self.st_encoder = PlaneResidualBlock(channels)
        self.uv_encoder = PlaneResidualBlock(channels)
        self.su_encoder = PlaneResidualBlock(channels)
        self.tv_encoder = PlaneResidualBlock(channels)
        self.cross_attn = FourPlaneCrossAttention(channels, num_heads=4, dropout=0.)
        if decoder_type != 'gs':
            raise ValueError("decoder_type='mlp' has been removed; only decoder_type='gs' is supported.")
        self.implicit_decoder = FourVolumeGaussianDecoder(
            channels,
            feat_unfold=feat_unfold,
            local_ensemble=local_ensemble,
            cell_decode=cell_decode,
            num_gaussians=num_gaussians,
            max_offset_px=max_offset_px,
            use_adaptive_offset=use_adaptive_offset,
            adaptive_offset_min_px=adaptive_offset_min_px,
            adaptive_offset_max_px=adaptive_offset_max_px,
            use_confidence_gate=use_confidence_gate,
            confidence_gate_blend=confidence_gate_blend,
            confidence_temperature=confidence_temperature
        )
        self.fusion = nn.Sequential(
            nn.Conv3d(channels * 4, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
        )

    def encode_st(self, x):
        b, c, u, v, h, w = x.shape
        plane = x.permute(0, 2, 3, 1, 4, 5).reshape(b * u * v, c, h, w)
        plane = self.st_encoder(plane)
        return plane.view(b, u, v, c, h, w).permute(0, 3, 1, 2, 4, 5).contiguous()

    def encode_uv(self, x):
        b, c, u, v, h, w = x.shape
        plane = x.permute(0, 4, 5, 1, 2, 3).reshape(b * h * w, c, u, v)
        plane = self.uv_encoder(plane)
        return plane.view(b, h, w, c, u, v).permute(0, 3, 4, 5, 1, 2).contiguous()

    def encode_su(self, x):
        b, c, u, v, h, w = x.shape
        plane = x.permute(0, 3, 5, 1, 4, 2).reshape(b * v * w, c, h, u)
        plane = self.su_encoder(plane)
        return plane.view(b, v, w, c, h, u).permute(0, 3, 5, 1, 4, 2).contiguous()

    def encode_tv(self, x):
        b, c, u, v, h, w = x.shape
        plane = x.permute(0, 2, 4, 1, 5, 3).reshape(b * u * h, c, w, v)
        plane = self.tv_encoder(plane)
        return plane.view(b, u, h, c, w, v).permute(0, 3, 1, 5, 2, 4).contiguous()

    def forward(self, x, angout=None):
        angout = self.angRes_out if angout is None else angout
        st = self.encode_st(x)
        uv = self.encode_uv(x)
        su = self.encode_su(x)
        tv = self.encode_tv(x)
        st, uv, su, tv = self.cross_attn(st, uv, su, tv)

        return self.implicit_decoder(st, uv, su, tv, angout, source=x)

    def forward_view(self, x, angout, u_idx, v_idx):
        st = self.encode_st(x)
        uv = self.encode_uv(x)
        su = self.encode_su(x)
        tv = self.encode_tv(x)
        st, uv, su, tv = self.cross_attn(st, uv, su, tv)
        return self.implicit_decoder.forward_view(st, uv, su, tv, angout, u_idx, v_idx)


class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)


def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


class InitFeaExtract(nn.Module):
    def __init__(self, channel):
        super(InitFeaExtract, self).__init__()
        self.FEconv = nn.Sequential(
            nn.Conv2d(1, channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x):
        b, n, _, h, w = x.shape
        x = x.contiguous().view(b * n, -1, h, w)
        buffer = self.FEconv(x)
        _, c, h, w = buffer.shape
        buffer = buffer.unsqueeze(1).contiguous().view(b, -1, c, h,
                                                       w)  # .permute(0,2,1,3,4)  # buffer_sv:  B, N, C, H, W

        return buffer


class Upsample(nn.Module):
    def __init__(self, channel, angular_in, angular_out):
        super(Upsample, self).__init__()
        self.an = angular_in
        self.an_out = angular_out
        self.angconv = nn.Sequential(
        )
        self.upsp = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=angular_in, stride=angular_in, padding=0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channel, channel * angular_out * angular_out, kernel_size=1, padding=0),
            nn.PixelShuffle(angular_out),
            nn.Conv2d(channel, 1, kernel_size=3, padding=1))

    def forward(self, x):
        b, n, c, h, w = x.shape
        x = x.contiguous().view(b, n, c, h * w)
        x = torch.transpose(x, 1, 3)
        x = x.contiguous().view(b * h * w, c, self.an, self.an)
        up_in = self.angconv(x)

        out = self.upsp(up_in)

        out = out.view(b, h * w, -1, self.an_out * self.an_out)
        out = torch.transpose(out, 1, 3)
        out = out.contiguous().view(b, self.an_out * self.an_out, -1, h, w)  # [N*81,c,h,w]
        return out


class SA_Epi_CrossAttention_Trans(nn.Module):
    def __init__(self, channels, emb_dim, MHSA_params):
        super(SA_Epi_CrossAttention_Trans, self).__init__()
        self.emb_dim = emb_dim
        self.sa_linear_in = nn.Linear(channels//2, emb_dim, bias=False)
        self.epi_linear_in = nn.Linear(channels//2, emb_dim, bias=False)
        self.sa_norm = nn.LayerNorm(emb_dim)
        self.epi_norm = nn.LayerNorm(emb_dim)
        self.attention = nn.MultiheadAttention(emb_dim,
                                               MHSA_params['num_heads'],
                                               MHSA_params['dropout'],
                                               bias=False)
        nn.init.kaiming_uniform_(self.attention.in_proj_weight, a=math.sqrt(5))
        self.attention.out_proj.bias = None
        self.attention.in_proj_bias = None
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim * 2, bias=False),
            nn.ReLU(True),
            nn.Dropout(MHSA_params['dropout']),
            nn.Linear(emb_dim * 2, emb_dim, bias=False),
            nn.Dropout(MHSA_params['dropout'])
        )
        self.linear_out = nn.Linear(emb_dim, channels//2, bias=False)

    def forward(self, buffer):
        # [_, _, n, v, w] = buffer.size()
        # b, c, u, h, v, w = buffer.shape
        b, c, u, v, h, w = buffer.shape

        # epi_token = rearrange(buffer, 'b c n v w -> (v w) (b n) c')
        token = buffer.permute(3, 5, 0, 2, 4, 1).reshape(v * w, b * u * h, c)
        sa_token = token[:, :, :c//2]
        epi_token = token[:, :, c//2:]

        epi_token_short_cut = epi_token

        sa_token = self.sa_linear_in(sa_token)
        epi_token = self.epi_linear_in(epi_token)

        sa_token_norm = self.sa_norm(sa_token)
        epi_token_norm = self.epi_norm(epi_token)
        sa_token = self.attention(query=sa_token_norm,
                                   key=epi_token_norm,
                                   value=sa_token,
                                   need_weights=False)[0] + sa_token

        sa_token = self.feed_forward(sa_token) + sa_token
        sa_token = self.linear_out(sa_token)

        buffer = torch.cat((sa_token, epi_token_short_cut), 2)
        # buffer = rearrange(epi_token, '(v w) (b n) c -> b c n v w', v=v, w=w, n=n)
        buffer = buffer.reshape(v, w, b, u, h, c).permute(2, 5, 3, 0, 4, 1).reshape(b, c, u, v, h, w)

        return buffer


class SA_Epi_Trans(nn.Module):
    def __init__(self, angRes, channels, MHSA_params):
        super(SA_Epi_Trans, self).__init__()
        self.angRes = angRes

        self.epi_trans = SA_Epi_CrossAttention_Trans(channels, channels * 2, MHSA_params)
        self.conv_1 = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False)
        )

    def forward(self, x):
        # [_, _, _, h, w] = x.size()
        b, c, n, h, w = x.size()

        u, v = self.angRes, self.angRes

        shortcut = x

        # EPI uh
        buffer = x.reshape(b, c, u, v, h, w).permute(0, 1, 3, 2, 5, 4)  # (b,c,v,u,w,h)
        buffer = self.conv_1(self.epi_trans(buffer).permute(0, 1, 3, 2, 5, 4).reshape(b, c, n, h, w)) + shortcut


        # EPI vw
        buffer = buffer.reshape(b, c, u, v, h, w)
        buffer = self.conv_1(self.epi_trans(buffer).reshape(b, c, n, h, w)) + shortcut
        # shortcut = buffer

        return buffer


class EpiXTrans(nn.Module):
    def __init__(self, channels, emb_dim, MHSA_params):
        super(EpiXTrans, self).__init__()
        self.emb_dim = emb_dim
        self.linear_in = nn.Linear(channels, emb_dim, bias=False)
        self.norm = nn.LayerNorm(emb_dim)
        self.attention = nn.MultiheadAttention(emb_dim,
                                               MHSA_params['num_heads'],
                                               MHSA_params['dropout'],
                                               bias=False)
        nn.init.kaiming_uniform_(self.attention.in_proj_weight, a=math.sqrt(5))
        self.attention.out_proj.bias = None
        self.attention.in_proj_bias = None
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim * 2, bias=False),
            nn.ReLU(True),
            nn.Dropout(MHSA_params['dropout']),
            nn.Linear(emb_dim * 2, emb_dim, bias=False),
            nn.Dropout(MHSA_params['dropout'])
        )
        self.linear_out = nn.Linear(emb_dim, channels, bias=False)

    ######### very important!!!
    def gen_mask(self, h: int, w: int, maxdisp: int = 6):  # when 30 Scenes Reflective Occlusion
        # def gen_mask(self, h: int, w: int, maxdisp: int=18):  # when HCI data
        attn_mask = torch.zeros([h, w, h, w])
        # k_h_left = k_h // 2
        # k_h_right = k_h - k_h_left
        # k_w_left = k_w // 2
        # k_w_right = k_w - k_w_left
        [ii, jj] = torch.meshgrid(torch.arange(h), torch.arange(w))

        for i in range(h):
            for j in range(w):
                temp = torch.zeros(h, w)
                temp[(ii - i).abs() * maxdisp >= (jj - j).abs()] = 1
                # temp[max(0, i - k_h_left):min(h, i + k_h_right), max(0, j - k_w_left):min(w, j + k_w_right)] = 1
                attn_mask[i, j, :, :] = temp

        # attn_mask = rearrange(attn_mask, 'a b c d -> (a b) (c d)')
        attn_mask = attn_mask.reshape(h * w, h * w)
        attn_mask = attn_mask.float().masked_fill(attn_mask == 0, float('-inf')).masked_fill(attn_mask == 1, float(0.0))

        return attn_mask

    def forward(self, buffer):
        # [_, _, n, v, w] = buffer.size()
        # b, c, u, h, v, w = buffer.shape
        b, c, u, v, h, w = buffer.shape
        # attn_mask = self.gen_mask(v, w, self.mask_field[0], self.mask_field[1]).to(buffer.device)
        attn_mask = self.gen_mask(v, w, ).to(buffer.device)

        # epi_token = rearrange(buffer, 'b c n v w -> (v w) (b n) c')
        epi_token = buffer.permute(3, 5, 0, 2, 4, 1).reshape(v * w, b * u * h, c)
        epi_token = self.linear_in(epi_token)

        epi_token_norm = self.norm(epi_token)
        epi_token = self.attention(query=epi_token_norm,
                                   key=epi_token_norm,
                                   value=epi_token,
                                   attn_mask=attn_mask,
                                   need_weights=False)[0] + epi_token

        epi_token = self.feed_forward(epi_token) + epi_token
        epi_token = self.linear_out(epi_token)
        # buffer = rearrange(epi_token, '(v w) (b n) c -> b c n v w', v=v, w=w, n=n)
        buffer = epi_token.reshape(v, w, b, u, h, c).permute(2, 5, 3, 0, 4, 1).reshape(b, c, u, v, h, w)

        return buffer


class EPIX_Trans(nn.Module):
    def __init__(self, angRes, channels, MHSA_params):
        super(EPIX_Trans, self).__init__()
        self.angRes = angRes

        self.epi_trans = EpiXTrans(channels, channels * 2, MHSA_params)
        self.conv_1 = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False)
        )

    def forward(self, x):
        # [_, _, _, h, w] = x.size()
        b, c, n, h, w = x.size()

        u, v = self.angRes, self.angRes

        shortcut = x

        # EPI uh
        buffer = x.reshape(b, c, u, v, h, w).permute(0, 1, 3, 2, 5, 4)  # (b,c,v,u,w,h)
        buffer = self.conv_1(self.epi_trans(buffer).permute(0, 1, 3, 2, 5, 4).reshape(b, c, n, h, w)) + shortcut


        # EPI vw
        buffer = buffer.reshape(b, c, u, v, h, w)
        buffer = self.conv_1(self.epi_trans(buffer).reshape(b, c, n, h, w)) + shortcut
        # shortcut = buffer

        return buffer


class C42_Conv(nn.Module):
    def __init__(self, ch, angRes):
        super(C42_Conv, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        S_ch, A_ch, E_ch, D_ch = ch, ch, ch // 2, ch // 2
        self.angRes = angRes
        self.spaconv = SpatialConv(ch)
        self.angconv = AngularConv(ch, angRes, A_ch)
        self.epiconv = EPiConv(ch, angRes, E_ch)
        self.dpiconv = EPiConv(ch, angRes, D_ch)
        self.fuse = nn.Sequential(
            nn.Conv3d(in_channels=S_ch + A_ch + E_ch + E_ch + D_ch + D_ch, out_channels=ch, kernel_size=1, stride=1,
                      padding=0, dilation=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(ch, ch, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), dilation=1))

    def forward(self, x):
        # b, n, c, h, w = x.shape
        b, c, n, h, w = x.shape
        an = int(math.sqrt(n))
        s_out = self.spaconv(x)
        a_out = self.angconv(x)

        epih_in = x.contiguous().view(b, c, an, an, h, w)  # b,c,u,v,h,w
        epih_out = self.epiconv(epih_in)

        # epiv_in = epih_in.permute(0,2,1,3,5,4)
        epiv_in = epih_in.permute(0, 1, 3, 2, 5, 4)  # b,c,v,u,w,h
        epiv_out = self.epiconv(epiv_in).reshape(b, -1, an, an, w, h).permute(0, 1, 3, 2, 5, 4).reshape(b, -1, n, h, w)

        dpih_in = epih_in.permute(0, 1, 3, 2, 4, 5)  # b,c,v,u,h,w
        dpih_out = self.dpiconv(dpih_in).reshape(b, -1, an, an, w, h).permute(0, 1, 3, 2, 4, 5).reshape(b, -1, n, h, w)

        dpiv_in = epih_in.permute(0, 1, 2, 3, 5, 4)  # b,c,u,v,w,h
        dpiv_out = self.dpiconv(dpiv_in).reshape(b, -1, an, an, w, h).permute(0, 1, 2, 3, 5, 4).reshape(b, -1, n, h, w)

        out = torch.cat((s_out, a_out, epih_out, epiv_out, dpih_out, dpiv_out), 1)
        out = self.fuse(out)

        return out + x  # out.contiguous().view(b,n,c,h,w) + x


class SA_Conv(nn.Module):
    def __init__(self, ch, angRes):
        super(SA_Conv, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        S_ch, A_ch, E_ch = ch, ch, ch // 2
        self.angRes = angRes
        self.spaconv = SpatialConv(ch)
        self.angconv = AngularConv(ch, angRes, A_ch)
        self.epiconv = EPiConv(ch, angRes, E_ch)
        self.SA_fuse = nn.Sequential(
            nn.Conv3d(in_channels=S_ch + A_ch, out_channels=ch, kernel_size=1, stride=1,
                      padding=0, dilation=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(ch, ch // 2, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), dilation=1))
        self.Epi_fuse = nn.Sequential(
            nn.Conv3d(in_channels= E_ch + E_ch, out_channels=ch, kernel_size=1, stride=1,
                      padding=0, dilation=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(ch, ch // 2, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), dilation=1))

    def forward(self, x):
        # b, n, c, h, w = x.shape
        b, c, n, h, w = x.shape
        an = int(math.sqrt(n))
        s_out = self.spaconv(x)
        a_out = self.angconv(x)

        sa_out = self.SA_fuse(torch.cat((s_out, a_out), 1))

        epih_in = x.contiguous().view(b, c, an, an, h, w)  # b,c,u,v,h,w
        epih_out = self.epiconv(epih_in)

        # epiv_in = epih_in.permute(0,2,1,3,5,4)
        epiv_in = epih_in.permute(0, 1, 3, 2, 5, 4)  # b,c,v,u,w,h
        epiv_out = self.epiconv(epiv_in).reshape(b, -1, an, an, w, h).permute(0, 1, 3, 2, 5, 4).reshape(b, -1, n, h, w)

        epi_out = self.Epi_fuse(torch.cat((epih_out, epiv_out), 1))

        out = torch.cat((sa_out, epi_out), 1)

        return out + x  # out.contiguous().view(b,n,c,h,w) + x


class SpatialConv(nn.Module):
    def __init__(self, ch):
        super(SpatialConv, self).__init__()
        self.spaconv_s = nn.Sequential(
            nn.Conv3d(in_channels=ch, out_channels=ch, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1),
                      dilation=(1, 1, 1)),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(in_channels=ch, out_channels=ch, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1),
                      dilation=(1, 1, 1)),
            nn.LeakyReLU(negative_slope=0.1, inplace=True))

    def forward(self, fm):
        return self.spaconv_s(fm)


class AngularConv(nn.Module):
    def __init__(self, ch, angRes, AngChannel):
        super(AngularConv, self).__init__()
        self.angconv = nn.Sequential(
            nn.Conv3d(ch * angRes * angRes, AngChannel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(AngChannel, AngChannel * angRes * angRes, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            # nn.PixelShuffle(angRes)
        )
        # self.an = angRes

    def forward(self, fm):
        b, c, n, h, w = fm.shape
        a_in = fm.contiguous().view(b, c * n, 1, h, w)
        out = self.angconv(a_in).view(b, -1, n, h, w)  # n == angRes * angRes
        return out


class EPiConv(nn.Module):
    def __init__(self, ch, angRes, EPIChannel):
        super(EPiConv, self).__init__()
        self.epi_ch = EPIChannel
        self.epiconv = nn.Sequential(
            nn.Conv3d(ch, EPIChannel, kernel_size=(1, angRes, angRes // 2 * 2 + 1), stride=1,
                      padding=(0, 0, angRes // 2), bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv3d(EPIChannel, angRes * EPIChannel, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0),
                      bias=False),  # ksize maybe (1,1,angRes//2*2+1) ?
            nn.LeakyReLU(0.1, True),
            # PixelShuffle1D(angRes),
        )
        # self.an = angRes

    def forward(self, fm):
        b, c, u, v, h, w = fm.shape

        epih_in = fm.permute(0, 1, 2, 4, 3, 5).reshape(b, c, u * h, v, w)
        epih_out = self.epiconv(epih_in)  # (b,self.epi_ch*v, u*h, 1, w)
        epih_out = epih_out.reshape(b, self.epi_ch, v, u, h, w).permute(0, 1, 3, 2, 4, 5).reshape(b, self.epi_ch, u * v,
                                                                                                  h, w)
        return epih_out


class PixelShuffle1D(nn.Module):
    def __init__(self, factor):
        super(PixelShuffle1D, self).__init__()
        self.factor = factor

    def forward(self, x):
        b, fc, h, w = x.shape
        c = fc // self.factor

        return x.view(b, c, h * self.factor, w)


class CascadedBlocks(nn.Module):
    '''
    Hierarchical feature fusion
    '''

    def __init__(self, n_blocks, channel, angRes):
        super(CascadedBlocks, self).__init__()
        self.n_blocks = n_blocks
        body = []
        for i in range(n_blocks):
            body.append(SA_Conv(channel, angRes))
        self.body = nn.Sequential(*body)
        # self.conv = nn.Conv2d(channel, channel, kernel_size = (3,3), stride = 1, padding = 1, dilation=1)
        self.conv = nn.Conv3d(channel, channel, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), dilation=1)

    def forward(self, x):
        buffer = x
        for i in range(self.n_blocks):
            buffer = self.body[i](buffer)
        buffer = self.conv(buffer) + x
        return buffer


# class CascadeC42Group(nn.Module):
#     def __init__(self, n_group, n_block, channels, angRes):
#         super(CascadeC42Group, self).__init__()
#         self.n_group = n_group
#         Groups = []
#         for i in range(n_group):
#             Groups.append(CascadedBlocks(n_block, channels, angRes))
#         self.Group = nn.Sequential(*Groups)
#         self.conv = nn.Conv3d(channels, channels, kernel_size = (1,3,3), stride = (1,1,1), padding = (0,1,1), dilation=1)
#
#     def forward(self, x):
#         buffer = x
#         for i in range(self.n_group):
#             buffer = self.Group[i](buffer)
#         buffer = self.conv(buffer)
#         return buffer + x

def LFsplit(data, angRes):
    b, _, H, W = data.shape
    h = int(H / angRes)
    w = int(W / angRes)
    data_sv = []
    for u in range(angRes):
        for v in range(angRes):
            data_sv.append(data[:, :, u * h:(u + 1) * h, v * w:(v + 1) * w])

    data_st = torch.stack(data_sv, dim=1)
    return data_st.permute(0, 2, 1, 3, 4)


def FormOutput(x_sv, angRes=None):
    x_sv = x_sv.permute(0, 2, 1, 3, 4)
    b, n, c, h, w = x_sv.shape
    angRes = int(math.sqrt(n)) if angRes is None else angRes
    out = []
    kk = 0
    for u in range(angRes):
        buffer = []
        for v in range(angRes):
            buffer.append(x_sv[:, kk, :, :, :])
            kk = kk + 1
        buffer = torch.cat(buffer, 3)
        out.append(buffer)
    out = torch.cat(out, 2)

    return out


if __name__ == "__main__":
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Net_CrossAttention(2, 7).to(device)

    #summary(net, input_size=(1, 1, 128, 128))
