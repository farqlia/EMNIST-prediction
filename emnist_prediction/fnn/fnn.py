import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics


class BaseFNN(pl.LightningModule):

    def __init__(self, n_classes: int, class_weights: torch.Tensor = None):
        super().__init__()

        self.n_classes = n_classes
        self.class_weights = class_weights

        self.train_f1_score = torchmetrics.classification.MulticlassF1Score(num_classes=n_classes)
        self.val_f1_score = torchmetrics.classification.MulticlassF1Score(num_classes=n_classes)

    def training_step(self, batch, batch_idx):
        x, y = batch
        predictions = self(x)
        loss = F.cross_entropy(predictions, y, weight=self.class_weights)
        self.log('train_loss', loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log('train_f1_score', self.train_f1_score, on_epoch=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        predictions = self(x)
        loss = F.cross_entropy(predictions, y)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True, on_epoch=True)
        self.log('val_f1_score', self.val_f1_score, on_epoch=True)
        return {'val_loss': loss}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        predictions = self(x)
        predicted_labels = torch.argmax(predictions, dim=-1)
        return predicted_labels

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class FNN(BaseFNN):

    def __init__(self, input_size, hidden_sizes, act_fn=F.relu, **kwargs):
        super().__init__(**kwargs)

        self.act_fn = act_fn

        layers = [nn.Linear(input_size, hidden_sizes[0])]
        layers += [nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]) for i in range(len(hidden_sizes) - 1)]
        layers += [nn.Linear(hidden_sizes[-1], self.n_classes)]
        self.linears = nn.ModuleList(layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for fc in self.linears[:-1]:
            x = self.act_fn(fc(x))
        x = self.linears[-1](x)
        return x


# More complex model with dropout regularization
class FNNReg(BaseFNN):

    def __init__(self, input_size, act_fn=F.relu, **kwargs):
        super().__init__(**kwargs)

        self.act_fn = act_fn
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.drop1 = nn.Dropout()
        self.fc3 = nn.Linear(512, 256)
        self.drop2 = nn.Dropout()
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, self.n_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.drop1(x)
        x = F.relu(self.fc3(x))
        x = self.drop2(x)
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x
