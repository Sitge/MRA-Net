from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from networks.module import MSWFModule
from networks.Vss import VSSBlock

nonlinearity = partial(F.relu, inplace=True)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels,
                                   bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.relu(x)
        x = self.pointwise(x)
        return x

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=(2, 3), keepdim=True)
        avg_pool = avg_pool.view(avg_pool.size(0), -1)

        excitation = self.fc1(avg_pool)
        excitation = F.relu(excitation)
        excitation = self.fc2(excitation)
        excitation = self.sigmoid(excitation).view(excitation.size(0), excitation.size(1), 1, 1)

        x = x * excitation
        return x


# AdaptiveReweightedAttentionBlock（ARA）
class ARA(nn.Module):
    def __init__(self, dim, qkv_bias=False, proj_drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.BatchNorm2d, mode='fc'):
        super(ARA, self).__init__()

        self.norm1 = norm_layer(dim)

        self.fc_h = nn.Conv2d(dim, dim, 1, 1, bias=qkv_bias)
        self.fc_w = nn.Conv2d(dim, dim, 1, 1, bias=qkv_bias)
        self.fc_c = nn.Conv2d(dim, dim, 1, 1, bias=qkv_bias)

        self.tfc_h = nn.Conv2d(2 * dim, dim, (1, 7), stride=1, padding=(0, 7 // 2), groups=dim, bias=False)
        self.tfc_w = nn.Conv2d(2 * dim, dim, (7, 1), stride=1, padding=(7 // 2, 0), groups=dim, bias=False)

        self.channel_attn = SEBlock(dim)

        self.spatial_attn = nn.Conv2d(dim, 1, kernel_size=1)

        self.alpha = nn.Parameter(torch.ones(1))

        self.reweight = DepthwiseSeparableConv(dim, dim * 3)

        self.proj = nn.Conv2d(dim, dim, 1, 1, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

        self.theta_h_conv = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, bias=True), nn.BatchNorm2d(dim), nn.ReLU())
        self.theta_w_conv = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, bias=True), nn.BatchNorm2d(dim), nn.ReLU())

        self.drop_path = nn.Identity() if drop_path <= 0. else nn.Dropout(drop_path)
        self.norm2 = norm_layer(dim)

    def forward(self, x):
        x1 = x
        x = self.norm1(x)
        B, C, H, W = x.shape

        theta_h = self.theta_h_conv(x)
        theta_w = self.theta_w_conv(x)

        x_h = self.fc_h(x)
        x_w = self.fc_w(x)
        x_h = torch.cat([x_h * torch.cos(theta_h), x_h * torch.sin(theta_h)], dim=1)
        x_w = torch.cat([x_w * torch.cos(theta_w), x_w * torch.sin(theta_w)], dim=1)

        h = self.tfc_h(x_h)
        w = self.tfc_w(x_w)
        c = self.fc_c(x)

        h = self.channel_attn(h)
        w = self.channel_attn(w)
        c = self.channel_attn(c)

        spatial_h = torch.sigmoid(self.spatial_attn(h))
        spatial_w = torch.sigmoid(self.spatial_attn(w))
        spatial_c = torch.sigmoid(self.spatial_attn(c))

        weighted_h = self.alpha * h * spatial_h
        weighted_w = (1 - self.alpha) * w * spatial_w
        weighted_c = self.alpha * c * spatial_c

        x = weighted_h + weighted_w + weighted_c

        a = F.adaptive_avg_pool2d(x, output_size=1)

        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(-1).unsqueeze(-1)

        x = weighted_h * a[0] + weighted_w * a[1] + weighted_c * a[2]

        x = self.proj(x)
        x = self.proj_drop(x)

        x = x1 + self.drop_path(x)

        return x



class PixelShuffleDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PixelShuffleDecoder, self).__init__()
        self.deconv = nn.Conv2d(in_channels, in_channels * 4, kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.deconv(x)
        x = self.pixel_shuffle(x)
        x = self.conv(x)
        return self.activation(self.norm(x))


class MRANet(nn.Module):
    def __init__(self, num_classes=1, in_channels=3):
        super(MRANet, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=False)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.attnBlock1 = ARA(filters[1], mode='fc')
        self.attnBlock2 = ARA(filters[2], mode='fc')
        self.attnBlock3 = ARA(filters[3], mode='fc')

        self.feature_fusion_1 = MSWFModule(in_channels=filters[0], out_channels=filters[0], scale_factor=1)
        self.feature_fusion_2 = MSWFModule(in_channels=filters[1], out_channels=filters[1], scale_factor=1)
        self.feature_fusion_3 = MSWFModule(in_channels=filters[2], out_channels=filters[2], scale_factor=1)

        self.decoder4 = PixelShuffleDecoder(filters[3], filters[2])
        self.decoder3 = PixelShuffleDecoder(filters[2], filters[1])
        self.decoder2 = PixelShuffleDecoder(filters[1], filters[0])
        self.decoder1 = PixelShuffleDecoder(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nn.GELU()
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nn.GELU()
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

        # VSSBlock
        self.vssBlock0 = VSSBlock(filters[0])
        self.vssBlock1 = VSSBlock(filters[1])
        self.vssBlock2 = VSSBlock(filters[2])
        self.vssBlock3 = VSSBlock(filters[3])

    def forward(self, x):
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)

        e1 = self.encoder1(x)

        e2 = self.encoder2(e1)
        e2 = self.attnBlock1(e2)
        e2 = e2.permute(0, 2, 3, 1)
        e2 = self.vssBlock1(e2)
        e2 = e2.permute(0, 3, 1, 2)

        e3 = self.encoder3(e2)
        e3 = self.attnBlock2(e3)
        e3 = e3.permute(0, 2, 3, 1)
        e3 = self.vssBlock2(e3)
        e3 = e3.permute(0, 3, 1, 2)

        e4 = self.encoder4(e3)
        e4 = self.attnBlock3(e4)
        e4 = e4.permute(0, 2, 3, 1)
        e4 = self.vssBlock3(e4)
        e4 = e4.permute(0, 3, 1, 2)

        d4 = self.decoder4(e4) + e3

        d4 = self.feature_fusion_3(d4)
        d4 = d4.permute(0, 2, 3, 1)
        d4 = self.vssBlock2(d4)
        d4 = d4.permute(0, 3, 1, 2)

        d3 = self.decoder3(d4) + e2

        d3 = self.feature_fusion_2(d3)
        d3 = d3.permute(0, 2, 3, 1)
        d3 = self.vssBlock1(d3)
        d3 = d3.permute(0, 3, 1, 2)

        d2 = self.decoder2(d3) + e1

        d2 = self.feature_fusion_1(d2)
        d2 = d2.permute(0, 2, 3, 1)
        d2 = self.vssBlock0(d2)
        d2 = d2.permute(0, 3, 1, 2)

        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return torch.sigmoid(out)


