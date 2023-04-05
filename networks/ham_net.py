import sys
import os
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torchvision.models import vgg16


model_urls = {
    "vgg16": "https://download.pytorch.org/models/vgg16-397923af.pth",
    "vgg16_bn": "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
}


class ham_net_base(nn.Module):
    def __init__(self, num_classes=3):
        super(ham_net_base, self).__init__()
        self.extra_convs = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, num_classes, 1),
        )
        self.classifier_b5 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.classifier_b4 = nn.Conv2d(512, num_classes, kernel_size=1)

        self._initialize_weights()

        model_vgg_baseline = vgg16(pretrained=True)
        self.features = model_vgg_baseline.features
        self.num_classes = num_classes

    def forward(self, x, CAM=False, size=None):
        x = self.features[:24](x)
        x_b4 = self.classifier_b4(x)

        x = self.features[24:](x)
        x_b5 = self.classifier_b5(x)

        x_b6 = self.extra_convs(x)

        logit_b6 = F.avg_pool2d(x_b6, kernel_size=(x_b6.size(2), x_b6.size(3)), padding=0)
        logit_b6 = logit_b6.view(-1, self.num_classes)

        logit_b5 = F.avg_pool2d(x_b5, kernel_size=(x_b5.size(2), x_b5.size(3)), padding=0)
        logit_b5 = logit_b5.view(-1, self.num_classes)

        logit_b4 = F.avg_pool2d(x_b4, kernel_size=(x_b4.size(2), x_b4.size(3)), padding=0)
        logit_b4 = logit_b4.view(-1, self.num_classes)

        if not CAM:
            return logit_b6, logit_b5, logit_b4
            # return logit_b5, logit_b4, logit_b3
        else:
            if size == None:
                return logit_b6, logit_b5,logit_b4,x_b6,x_b5,x_b4
            else:
                x_b6 = self.cam_normalize(x_b6, size)
                x_b5 = self.cam_normalize(x_b5, size)
                x_b4 = self.cam_normalize(x_b4, size)
                return logit_b6, logit_b5, logit_b4, x_b6, x_b5, x_b4

    def cam_normalize(self, cam, size):
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=size, mode="bilinear", align_corners=True)
        cam = cam / (F.adaptive_max_pool2d(cam, 1) + 1e-5)
        return cam

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def get_parameter_groups(self):
        groups = ([], [], [], [])

        for name, value in self.named_parameters():

            if "extra" in name:
                if "weight" in name:
                    groups[2].append(value)
                else:
                    groups[3].append(value)
            else:
                if "weight" in name:
                    groups[0].append(value)
                else:
                    groups[1].append(value)
        return groups


class ham_net(ham_net_base):
    def __init__(self, num_classes=3):
        super(ham_net, self).__init__(num_classes)
        del self.features[30]
        del self.features[23]