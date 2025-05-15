import torch
import torch.nn as nn
import torch.nn.functional as F


def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]

    min_height = min(x1.size(2), x2.size(2), x3.size(2), x4.size(2))
    min_width = min(x1.size(3), x2.size(3), x3.size(3), x4.size(3))

    x1 = x1[:, :, :min_height, :min_width]
    x2 = x2[:, :, :min_height, :min_width]
    x3 = x3[:, :, :min_height, :min_width]
    x4 = x4[:, :, :min_height, :min_width]

    LL = x1 + x2 + x3 + x4
    HL = -x1 - x2 + x3 + x4
    LH = -x1 + x2 - x3 + x4
    HH = x1 - x2 - x3 + x4

    return LL, HL, LH, HH


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)





class MultiScaleFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleFeatureExtractor, self).__init__()

        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv7x7 = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3)


    def forward(self, x):

        x1 = self.conv1x1(x)
        x2 = self.conv3x3(x)
        x3 = self.conv5x5(x)
        x4 = self.conv7x7(x)



        features = torch.cat([x1, x2, x3, x4], dim=1)

        return features





class WaveletAttention(nn.Module):
    def __init__(self):
        super(WaveletAttention, self).__init__()
        self.dwt = DWT()

    def forward(self, x):

        dwt_result = self.dwt(x)
        cA, cH, cV, cD = dwt_result


        return cA, cH, cV, cD





class FeatureMapping(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureMapping, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=1)
        self.silu = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.silu(self.conv1(x))
        x = self.conv2(x)
        return x



class FeatureModulation(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=1):
        super(FeatureModulation, self).__init__()

        self.mapping = FeatureMapping(in_channels*16, out_channels*4)
        self.scale_factor = scale_factor

    def forward(self, multi_scale_features, recombined_features):
        modulation_params = self.mapping(recombined_features)

        if self.scale_factor > 1:
            modulation_params = F.interpolate(modulation_params, scale_factor=self.scale_factor, mode='bilinear')

        desired_size = (multi_scale_features.size(2), multi_scale_features.size(3))
        modulation_params = F.interpolate(modulation_params, size=desired_size, mode='bilinear', align_corners=False)
        modulated_feature_map = multi_scale_features * modulation_params
        return modulated_feature_map


class MSWFModule(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super(MSWFModule, self).__init__()

        self.multi_scale_extractor = MultiScaleFeatureExtractor(in_channels, in_channels)

        self.wavelet_attention = WaveletAttention()

        self.feature_modulation = FeatureModulation(in_channels, out_channels, scale_factor=1)

        self.fusion_conv = nn.Conv2d(in_channels*4, out_channels, kernel_size=1)

        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):

        multi_scale_features = self.multi_scale_extractor(x)

        attn_cA, attn_cH, attn_cV, attn_cD = self.wavelet_attention(multi_scale_features)

        recombined_features = torch.cat((attn_cA, attn_cH, attn_cV, attn_cD), dim=1)

        modulated_large_scale_features = self.feature_modulation(multi_scale_features, recombined_features)

        fused_features = self.fusion_conv(modulated_large_scale_features)

        return fused_features


