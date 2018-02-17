from deeproof.common import PRETRAINED_DIR
from torch import nn, ones
from torchvision import models
from torch.nn.init import kaiming_normal
from torch import np
import torch
import torch.nn.functional as F


# ResNet fine-tuning
class ResNet50(nn.Module):
    # We use ResNet weights from PyCaffe.
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()

        # Loading ResNet arch from PyTorch and weights from Pycaffe
        original_model = models.resnet50(pretrained=False)
        original_model.load_state_dict(torch.load((PRETRAINED_DIR / 'resnet50.pth').as_posix()))

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


class ResNet101(nn.Module):
    # We use ResNet weights from PyCaffe.
    def __init__(self, num_classes):
        super(ResNet101, self).__init__()

        # Loading ResNet arch from PyTorch and weights from Pycaffe
        original_model = models.resnet101(pretrained=False)
        original_model.load_state_dict(torch.load((PRETRAINED_DIR / 'resnet101.pth').as_posix()))

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


class ResNet152(nn.Module):
    # We use ResNet weights from PyCaffe.
    def __init__(self, num_classes):
        super(ResNet152, self).__init__()

        # Loading ResNet arch from PyTorch and weights from Pycaffe
        original_model = models.resnet152(pretrained=False)
        original_model.load_state_dict(torch.load((PRETRAINED_DIR / 'resnet152.pth').as_posix()))

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
