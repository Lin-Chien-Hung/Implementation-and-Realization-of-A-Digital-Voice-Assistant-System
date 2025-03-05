import torch
import torch.nn as nn

# from resnest.torch import resnest50
from .ResNeSt.resnest.torch import resnest50
from . import resnet_256 as resnet
from . import resnet_all as resnet_a
from torch.nn import Parameter


class BackgroundResnet(nn.Module):
    def __init__(self, num_classes, backbone="resnest50"):
        model_num_class = 512  # resnet34 為 256, resnest50 = 512
        super(BackgroundResnet, self).__init__()
        self.backbone = backbone
        # copying modules from pretrained models
        if backbone == "resnet50":
            self.pretrained = resnet.resnet50(pretrained=False)
        elif backbone == "resnet101":
            self.pretrained = resnet.resnet101(pretrained=False)
        elif backbone == "resnet34":
            self.pretrained = resnet.resnet34(pretrained=False)
        elif backbone == "resnext50_32x4d":
            self.pretrained = resnet_a.resnext50_32x4d(
                pretrained=False, num_classes=model_num_class
            )
        elif backbone == "resnet340":
            self.pretrained = resnet_a.resnet34(pretrained=False)
        elif backbone == "resnest50":
            self.pretrained = resnest50(pretrained=False, num_classes=model_num_class)
        else:
            raise RuntimeError("unknown backbone: {}".format(backbone))

        self.weight = Parameter(torch.Tensor(num_classes, model_num_class))
        nn.init.xavier_uniform_(self.weight)

        self.relu = nn.ReLU()

    def forward(self, x):
        # input x: minibatch x 1 x 40 x 40
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)

        x = self.pretrained.layer1(x)
        x = self.pretrained.layer2(x)
        x = self.pretrained.layer3(x)
        x = self.pretrained.layer4(x)

        x = self.pretrained.avgpool(x)
        x = torch.flatten(x, 1)
        if self.pretrained.drop:
            x = self.pretrained.drop(x)
        x = self.pretrained.fc(x)  # x = spk_embedding

        return x
