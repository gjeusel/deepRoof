from deeproof.common import PRETRAINED_DIR
from torch import nn, ones
from torchvision import models
from torch.nn.init import kaiming_normal
from torch import np
import torch
import torch.nn.functional as F


# ResNet fine-tuning
class ResNet(nn.Module):
    # We use ResNet weights from PyCaffe.
    def __init__(self, num_classes, resnet=34, pretrained=False):
        super(ResNet, self).__init__()

        # Loading ResNet arch from PyTorch and weights from Pycaffe
        if resnet == 18:
            original_model = models.resnet18(pretrained=pretrained)
        elif resnet == 34:
            original_model = models.resnet34(pretrained=pretrained)
        elif resnet == 50:
            original_model = models.resnet50(pretrained=pretrained)
        elif resnet == 101:
            original_model = models.resnet101(pretrained=pretrained)
        elif resnet == 152:
            original_model = models.resnet152(pretrained=pretrained)
        else:
            raise ValueError('ResNet n°{} is unknown.'.format(resnet))

        fpath = PRETRAINED_DIR / ('resnet' + str(resnet) + '.pth')
        original_model.load_state_dict(torch.load(fpath.as_posix()))

        # Everything except the last linear layer
        self.features = nn.Sequential(*list(original_model.children())[:-1])

        # Get number of features of last layer
        num_feats = original_model.fc.in_features

        # Plug our classifier
        self.classifier = nn.Sequential(
            nn.Linear(num_feats, num_classes)
        )

        # Init of last layer
        for m in self.classifier:
            kaiming_normal(m.weight)

        # Freeze those weights
        # for p in self.features.parameters():
        #     p.requires_grad = False

    def forward(self, x):
        f = self.features(x)
        f = f.view(f.size(0), -1)
        y = self.classifier(f)
        return y


def get_conv_output(shape, layer):
    bs = 1
    x = torch.autograd.Variable(torch.rand(bs, *shape))
    out = layer(x)
    return out.size()[1:]


class ShortNet(nn.Module):
    """Convolutional Neural Network short layers."""

    def __init__(self, input_shape):
        """
        :param tuple input_shape: shape of inputs tensor.
            example: (3, 96, 96) with 3 for RGB.
        """
        super(ShortNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(),
            # nn.BatchNorm2d(32),
            nn.Conv2d(32, 16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2))

        shape = get_conv_output(input_shape, self.layer1)
        shape = get_conv_output(shape, self.layer2)

        self.fc1 = nn.Linear(shape[0] * shape[1] *
                             shape[2], 120)  # fully connected
        self.fc2 = nn.Linear(120, 84)  # fully connected
        self.fc3 = nn.Linear(84, 4)  # fully connected

    def forward(self, x, training=True):
        out = self.layer1(x)
        out = nn.functional.dropout2d(out, 0.15)
        out = self.layer2(out)
        out = nn.functional.dropout2d(out, 0.20)

        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out
