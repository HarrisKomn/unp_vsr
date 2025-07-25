import torch
import torch.nn as nn

from mmcv.runner import load_checkpoint
from mmedit.models.common import (PixelShufflePack, flow_warp)

from mmedit.utils import get_root_logger
from basicsr.archs.spynet_arch import SpyNet as SpyNetBSR
from mmengine.model import BaseModule


class InstanceNormAlternative(nn.InstanceNorm2d):
    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(inp)
        desc = 1 / (inp.var(axis=[2, 3], keepdim=True, unbiased=False) + self.eps) ** 0.5
        retval = (inp - inp.mean(axis=[2, 3], keepdim=True)) * desc
        return retval


class BasicVSRNet(BaseModule):

    def __init__(self, mid_channels=128, num_blocks=30, spynet_pretrained=None):

        super().__init__()

        self.mid_channels = mid_channels

        # optical flow network for feature alignment
        self.spynet = SpyNetBSR(spynet_pretrained)
        self.spynet.train(False)
        for param in self.spynet.parameters():
            param.requires_grad = False

        norm_layer=InstanceNormAlternative
        use_bias=True
        padding_type = 'reflect'
        use_dropout = True


        backward_resblocks = [
            nn.Conv2d(mid_channels + 1, mid_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True), # extra addition
            ResnetBlock(mid_channels, mid_channels, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                        use_bias=use_bias),
            ResnetBlock(mid_channels, mid_channels, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                        use_bias=use_bias),
            ResnetBlock(mid_channels, mid_channels, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                        use_bias=use_bias),
            ResnetBlock(mid_channels, mid_channels, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                        use_bias=use_bias),
            ResnetBlock(mid_channels, mid_channels, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                        use_bias=use_bias),
            ResnetBlock(mid_channels, mid_channels, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                        use_bias=use_bias),
            ResnetBlock(mid_channels, mid_channels, padding_type=padding_type, norm_layer=norm_layer,
                        use_dropout=use_dropout,
                        use_bias=use_bias),
            ResnetBlock(mid_channels, mid_channels, padding_type=padding_type, norm_layer=norm_layer,
                        use_dropout=use_dropout,
                        use_bias=use_bias),
            ResnetBlock(mid_channels, mid_channels, padding_type=padding_type, norm_layer=norm_layer,
                        use_dropout=use_dropout,
                        use_bias=use_bias),
            ResnetBlock(mid_channels, mid_channels, padding_type=padding_type, norm_layer=norm_layer,
                        use_dropout=use_dropout,
                        use_bias=use_bias)
        ]

        self.backward_resblocks = nn.Sequential(*backward_resblocks)

        forward_resblocks = [
            nn.Conv2d(mid_channels + 1, mid_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True), # extra addition

            ResnetBlock(mid_channels, mid_channels, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                        use_bias=use_bias),
            ResnetBlock(mid_channels, mid_channels, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                        use_bias=use_bias),
            ResnetBlock(mid_channels, mid_channels, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                        use_bias=use_bias),
            ResnetBlock(mid_channels, mid_channels, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                        use_bias=use_bias),
            ResnetBlock(mid_channels, mid_channels, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                        use_bias=use_bias),
            ResnetBlock(mid_channels, mid_channels, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                        use_bias=use_bias),
            ResnetBlock(mid_channels, mid_channels, padding_type=padding_type, norm_layer=norm_layer,
                        use_dropout=use_dropout,
                        use_bias=use_bias),
            ResnetBlock(mid_channels, mid_channels, padding_type=padding_type, norm_layer=norm_layer,
                        use_dropout=use_dropout,
                        use_bias=use_bias),
            ResnetBlock(mid_channels, mid_channels, padding_type=padding_type, norm_layer=norm_layer,
                        use_dropout=use_dropout,
                        use_bias=use_bias),
            ResnetBlock(mid_channels, mid_channels, padding_type=padding_type, norm_layer=norm_layer,
                        use_dropout=use_dropout,
                        use_bias=use_bias)
        ]
        self.forward_resblocks = nn.Sequential(*forward_resblocks)


        ## downsampling
        #self.backward_downsampling = nn.Conv2d(mid_channels + 1, mid_channels + 1, kernel_size=3, stride=2, padding=1)
        self.backward_downsampling = nn.Conv2d(mid_channels + 1, mid_channels + 1, kernel_size=3, stride=2, padding=1)
        self.forward_downsampling = nn.Conv2d(mid_channels + 1, mid_channels + 1, kernel_size=3, stride=2, padding=1)
        self.downsampling = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)

        # upsample
        self.fusion = nn.Conv2d(
            mid_channels*2, mid_channels, 1, 1, 0, bias=True)
        self.upsample1 = PixelShufflePack(
            mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(
            mid_channels, 64, 2, upsample_kernel=3)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 1, 3, 1, 1)
        self.img_upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.tanh = nn.Tanh()



    def check_if_mirror_extended(self, lrs):


        self.is_mirror_extended = False
        if lrs.size(1) % 2 == 0:
            lrs_1, lrs_2 = torch.chunk(lrs, 2, dim=1)
            if torch.norm(lrs_1 - lrs_2.flip(1)) == 0:
                self.is_mirror_extended = True

    def compute_flow(self, lrs):

        lrs = lrs.repeat(1,1,3, 1, 1)
        n, t, c, h, w = lrs.size()

        lrs_1 = lrs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lrs_2 = lrs[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(lrs_1, lrs_2).view(n, t - 1, 2, h, w)

        if self.is_mirror_extended:  # flows_forward = flows_backward.flip(1)
            flows_forward = None
        else:
            flows_forward = self.spynet(lrs_2, lrs_1).view(n, t - 1, 2, h, w)
        return flows_forward, flows_backward



    def forward(self, lrs, encode_only=False):

        n, t, c, h, w = lrs.size()
        assert h >= 64 and w >= 64, (
            'The height and width of inputs should be at least 64, '
            f'but got {h} and {w}.')

        if encode_only:
            rec_feats = []
            rec_feats.append([lrs[:, 0, :, :, :]]) # Training_1_seqs
            rec_feats.append([lrs[:, 1, :, :, :]]) # Training_2_seqs
            rec_feats.append([lrs[:, 2, :, :, :]]) # Training_3_seqs
            rec_feats.append([lrs[:, 3, :, :, :]]) # Training_4_seqs
            rec_feats.append([lrs[:, 4, :, :, :]]) # Training_5_seqs

        # check whether the input is an extended sequence
        self.check_if_mirror_extended(lrs)
        lrs = self.downsampling(lrs.view(-1, c, h, w))

        h,w = h//2, w//2
        lrs = lrs.view(-1, t, c, h, w)

        # compute optical flow
        lrs_flow = (0.5*255.*lrs+0.5*255.)/255.
        flows_forward, flows_backward = self.compute_flow(lrs_flow)

        # backward-time propagation
        outputs = []
        feat_prop = lrs.new_zeros(n, self.mid_channels, h, w)
        for i in range(t - 1, -1, -1):
            if i < t - 1:  # no warping required for the last timestep
                flow = flows_backward[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1)) # bidirectional

            feat_prop = torch.cat([lrs[:, i, :, :, :], feat_prop], dim=1)
            feat_prop = self.backward_resblocks(feat_prop)

            outputs.append(feat_prop)
        outputs = outputs[::-1]

        # forward-time propagation and upsampling
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, t):
            lr_curr = lrs[:, i, :, :, :]
            if i > 0:  # no warping required for the first timestep
                if flows_forward is not None:
                    flow = flows_forward[:, i - 1, :, :, :]
                else:
                    flow = flows_backward[:, -i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

            feat_prop = torch.cat([lr_curr, feat_prop], dim=1)

            layers = [0, 2, 4]
            if (encode_only):
                if -1 in layers:
                    layers.append(len(self.forward_resblocks))
                if len(layers) > 0:
                    feat = feat_prop
                    for layer_id, layer in enumerate(self.forward_resblocks):
                        feat = layer(feat)
                        if layer_id in layers:
                            if layer_id == 0:
                                rec_feats[i].append(lrs[:, i, :, :, :])
                            rec_feats[i].append(feat)
                        else:

                            pass

                    feat_prop = feat

            else:

                feat_prop = self.forward_resblocks(feat_prop)

                # upsampling given the backward and forward features
                out = torch.cat([outputs[i], feat_prop], dim=1)
                out = self.lrelu(self.fusion(out))
                out = self.lrelu(self.upsample2(out))
                out = self.conv_last(out)
                out = self.tanh(out)
                outputs[i] = out

        if (encode_only):

            # WITH ALL LAYERS - 5 LAYERS
            total_feats = [rec_feats[-1][0], rec_feats[-1][1], rec_feats[-1][2], rec_feats[-1][3], rec_feats[-1][4]]
            neighbour_feats = [rec_feats[0], rec_feats[1], rec_feats[2], rec_feats[3], rec_feats[4]]

            # WITHOUT THE FIRST LAYER  - 4 LAYERS
            # total_feats = [rec_feats[-1][1], rec_feats[-1][2], rec_feats[-1][3], rec_feats[-1][4]]
            # neighbour_feats = [rec_feats[1], rec_feats[2], rec_feats[3], rec_feats[4]]


            # WITHOUT THE LAST 2- LAYERS  - 3 LAYERS
            # total_feats = [rec_feats[-1][0], rec_feats[-1][1], rec_feats[-1][2]]
            # neighbour_feats = [rec_feats[0], rec_feats[1], rec_feats[2]]

            return total_feats , neighbour_feats

        return torch.stack(outputs, dim=1)


    def init_weights(self, pretrained=None, strict=True):

        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')


class ResnetBlock(nn.Module):

    def __init__(self, input_dim, output_dim, padding_type, norm_layer, use_dropout, use_bias):

        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(input_dim, output_dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, input_dim, output_dim, padding_type, norm_layer, use_dropout, use_bias):

        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(output_dim), nn.ReLU(True)]

        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(output_dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out




