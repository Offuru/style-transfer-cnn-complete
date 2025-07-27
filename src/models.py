import torch
import torch.nn as nn
from torchvision import models


def get_vgg19_model():
    vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).eval()
    return vgg


def gram_matrix(x):
    _, c, h, w = x.size()
    features = x.view(c, h * w)
    G = torch.mm(features, features.t())
    return G, c, h * w


class NormalizationLayer(nn.Module):
    def __init__(self, mean, std):
        super(NormalizationLayer, self).__init__()
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)
    def forward(self, x):
        return ... # Normalize the input tensor using the mean and standard deviation


class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, x):
        self.loss = nn.functional.mse_loss(...) # loss = MSE(x, target) / 2
        return x


class StyleLoss(nn.Module):
    def __init__(self, target):
        super(StyleLoss, self).__init__()
        self.target, _, _ = gram_matrix(target)
        self.target = self.target.detach()

    def forward(self, x):
        G, channels, features = ... # Compute the Gram matrix for the input tensor
        self.loss = nn.functional.mse_loss(...) # loss = MSE(G, target) / (4 * channels^2 * features^2)
        return x
