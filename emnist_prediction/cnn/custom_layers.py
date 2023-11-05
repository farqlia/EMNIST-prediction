import torch.nn as nn


class CustomConv2d(nn.Module):

    def __init__(self, n_channels_in, n_channels_out, kernel_size, pad,
                 use_pooling=False, stride=1, drop=None, batch_norm=True,
                 activ=nn.ReLU):
        super(CustomConv2d, self).__init__()
        layers = [nn.Conv2d(n_channels_in, n_channels_out, kernel_size, stride=stride,
                            padding=pad)]
        if use_pooling:
            layers.append(nn.MaxPool2d(kernel_size=kernel_size))
        if activ:
            layers.append(activ())
        if batch_norm:
            layers.append(nn.BatchNorm2d(n_channels_out))
        if drop:
            layers.append(nn.Dropout2d(drop))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Flatten(nn.Module):

    def __init__(self, keep_batch_dim=True):
        super(Flatten, self).__init__()
        self.keep_batch_dim = keep_batch_dim

    def forward(self, x):
        if self.keep_batch_dim:
            return x.view(x.shape[0], -1)
        return x.view(-1)
