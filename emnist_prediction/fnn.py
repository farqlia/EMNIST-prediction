import torch
import torch.nn as nn
import torchmetrics 
import torch.nn.functional as F
import pytorch_lightning as pl

class FNN(pl.LightningModule):

    def __init__(self, input_size, n_classes, hidden_sizes, class_weights=None, act_fn=F.relu):
        super().__init__()

        self.train_f1_score = torchmetrics.classification.MulticlassF1Score(num_classes=n_classes)
        self.val_f1_score = torchmetrics.classification.MulticlassF1Score(num_classes=n_classes)
        self.class_weights = class_weights
        self.act_fn = act_fn

        layers = [nn.Linear(input_size, hidden_sizes[0])]
        layers += [nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]) for i in range(len(hidden_sizes) - 1)]

        layers += [nn.Linear(hidden_sizes[-1], n_classes)]

        self.linears = nn.ModuleList(layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for fc in self.linears:
            x = self.act_fn(fc(x))
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        predictions = self(x)
        loss = F.cross_entropy(predictions, y, weight=self.class_weights)
        predicted_labels = torch.argmax(predictions, axis=-1)
        self.log('train_loss', loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log('train_f1_score', self.train_f1_score, on_epoch=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        predictions = self(x)
        loss = F.cross_entropy(predictions, y)
        predicted_labels = torch.argmax(predictions, axis=-1)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log('val_f1_score', self.val_f1_score, on_epoch=True)
        return {'val_loss': loss}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        predictions = self(x)
        predicted_labels = torch.argmax(predictions, axis=-1)
        return predicted_labels

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer