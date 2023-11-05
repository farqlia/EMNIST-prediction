from typing import List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from emnist_prediction.cnn.custom_layers import CustomConv2d, Flatten


class BaseCNN(pl.LightningModule):

    def __init__(self, class_weights=None, act_fn=F.relu):
        super().__init__()

        self.act_fn = act_fn
        self.class_weights = class_weights

    def training_step(self, batch, batch_idx):
        x, y = batch
        predictions = self(x)
        loss = F.cross_entropy(predictions, y, weight=self.class_weights)
        self.log('train_loss', loss, prog_bar=True, sync_dist=True, on_epoch=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        predictions = self(x)
        loss = F.cross_entropy(predictions, y)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True, on_epoch=True)
        return {'val_loss': loss}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        predictions = self(x)
        predicted_labels = torch.argmax(predictions, dim=-1)
        return predicted_labels

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class DeepCNN(BaseCNN):

    def __init__(self, convolutional_layers: List[CustomConv2d],
                 linear_layers: List[nn.Linear], **kwargs):
        super(DeepCNN, self).__init__(**kwargs)

        self.conv_layers = nn.Sequential(*convolutional_layers)
        self.flatten = Flatten()
        self.lin_layers = nn.ModuleList(linear_layers)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        # x = self.dropout(x)
        for lin_layer in self.lin_layers[:-1]:
            x = self.act_fn(lin_layer(x))
        x = self.lin_layers[-1](x)
        return x
