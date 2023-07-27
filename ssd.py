import torch

from default_box import DefaultBox
from layer.function.detection import Detect
from layer.module.l2norm import L2Norm
import config
from lib import *


def vgg():
    layers = []
    in_channels = 3
    layer_cfgs = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'CM', 512, 512, 512, 'M', 512, 512, 512]
    re_lu = nn.ReLU(inplace=True)
    for cfg in layer_cfgs:
        if cfg == 'M':  # floor
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif cfg == 'CM':  # ceiling
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels=in_channels, out_channels=cfg, kernel_size=3, padding=1)
            layers += [conv2d, re_lu]
            in_channels = cfg

    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1)
    layers += [pool5, conv6, re_lu, conv7, re_lu]
    return nn.ModuleList(layers)


def extras():
    layers = []
    in_channel = 1024
    in_channels = [256, 512, 128, 256, 128, 256, 128, 256]
    kernel_sizes = [1, 3, 1, 3, 1, 3, 1, 3]
    strides = [1, 2, 1, 2, 1, 1, 1, 1]
    padding = [0, 1, 0, 1, 0, 0, 0, 0]
    for i in range(8):
        layers += [nn.Conv2d(in_channels=in_channel, out_channels=in_channels[i], kernel_size=kernel_sizes[i],
                             stride=strides[i], padding=padding[i])]
        in_channel = in_channels[i]
    return nn.ModuleList(layers)


def loc_conf(num_classes=21, in_channels=(512, 1024, 512, 256, 256, 256), bounding_box_ratio_num=(4, 6, 6, 6, 4, 4)):
    loc_layers = []
    conf_layers = []

    for i in range(6):
        loc_layers += [nn.Conv2d(in_channels=in_channels[i], out_channels=bounding_box_ratio_num[i] * 4,
                                 kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(in_channels=in_channels[i], out_channels=bounding_box_ratio_num[i] * num_classes,
                                  kernel_size=3, padding=1)]

    return nn.ModuleList(loc_layers), nn.ModuleList(conf_layers)


class SSD(nn.Module):
    def __init__(self, phase, cfgs):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = cfgs["num_class"]

        # main module
        self.vgg = vgg()
        self.extras = extras()
        self.loc, self.conf = loc_conf(num_classes=cfgs["num_class"], bounding_box_ratio_num=cfgs["bbox_aspect_num"])
        self.l2_norm = L2Norm()

        # default box
        default_box = DefaultBox(cfgs)
        self.default_boxes = default_box.default_box()
        if phase == "inference":
            self.detect = Detect()

    def forward(self, x):
        source = list()
        loc = list()
        conf = list()
        for k in range(23):
            x = self.vgg[k](x)
        source_1 = self.l2_norm(x)
        source.append(source_1)
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        # source 2
        source.append(x)
        # source 3 -> 6
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 != 0:
                source.append(x)
        for (x, l, c) in zip(source, self.loc, self.conf):
            # aspect_ratio_num = 4 or 6
            # (batch_size, 4 * aspect_ratio_num, feature_map_height, feature_map_weight)
            # -> (batch_size, feature_map_height, feature_map_weight, 4 * aspect_ratio_num)
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([lo.view(lo.size(0), -1) for lo in loc], 1)  # (batch_size, 34928) 4*8732
        conf = torch.cat([cf.view(cf.size(0), -1) for cf in conf], 1)  # (batch_size, 8732*21)
        loc = loc.view(loc.size(0), -1, 4)  # (batch_size, 8732, 4)
        conf = conf.view(conf.size(0), -1, self.num_classes)  # (batch_size, 8732, 21)

        output = (loc, conf, self.default_boxes)
        if self.phase == "inference":
            with torch.no_grad():
                return self.detect(output[0], output[1], output[2])
        else:
            return output


if __name__ == '__main__':
    # vgg = vgg()
    # print(vgg)
    ssd = SSD(phase="train", cfgs=config.cfgs)
    print(ssd)
