# get_ipython().system('pip install torch pytorch-lightning torchvision matplotlib  scikit-learn')


import torch
import pytorch_lightning as p
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on {device}")


from torchvision import datasets
from torchvision.transforms import ToTensor

train_data = datasets.MNIST(
    root="data", train=True, transform=ToTensor(), download=True
)

test_data = datasets.MNIST(root="data", train=False, transform=ToTensor())


from sklearn.model_selection import train_test_split


train_indices, val_indices, _, _ = train_test_split(
    range(len(train_data)),
    train_data.targets,
    stratify=train_data.targets,
    test_size=0.2,
)


train_split = torch.utils.data.Subset(train_data, train_indices)
val_split = torch.utils.data.Subset(train_data, val_indices)


from torchvision import datasets
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
import torch


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.batch_size = 32
        self.train_split = None
        self.val_split = None
        self.test_split = None

    def prepare_data(self):
        datasets.MNIST(root="data", train=True, transform=ToTensor(), download=True)
        datasets.MNIST(root="data", train=False, transform=ToTensor(), download=True)

    def setup(self, stage=None):
        if stage == "fit":
            self.train_dataset = datasets.MNIST(
                root="data", train=True, transform=ToTensor(), download=False
            )
            self.test_dataset = datasets.MNIST(
                root="data", train=False, transform=ToTensor(), download=False
            )

            self.train_indices, self.val_indices, _, _ = train_test_split(
                range(len(self.train_dataset)),
                self.train_dataset.targets,
                stratify=self.train_dataset.targets,
                test_size=0.2,
            )

            self.train_split = torch.utils.data.Subset(
                self.train_dataset, self.train_indices
            )
            self.val_split = torch.utils.data.Subset(
                self.train_dataset, self.val_indices
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_split, batch_size=self.batch_size)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_split, batch_size=self.batch_size)

    def test_dataloader(self: p.LightningDataModule):
        self.test_indices = range(len(self.test_dataset))
        self.test_split = torch.utils.data.Subset(self.test_dataset, self.test_indices)
        return torch.utils.data.DataLoader(self.test_split, batch_size=self.batch_size)


import torch.nn as nn
import torchmetrics


class CNN(p.LightningModule):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc1 = nn.Linear(32 * 7 * 7, 10)

        self.train_acc = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=10
        )
        self.valid_acc = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=10
        )
        self.test_acc = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=10
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = x.view(x.size(0), -1)

        out = self.fc1(x)

        return out

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = nn.CrossEntropyLoss()(y_pred, y)
        self.train_acc(y_pred, y)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=False)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = nn.CrossEntropyLoss()(y_pred, y)
        self.valid_acc(y_pred, y)
        self.log("val_acc", self.valid_acc, on_step=True, on_epoch=True)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = nn.CrossEntropyLoss()(y_pred, y)
        self.log("test_loss", loss)
        self.test_acc(y_pred, y)
        self.log("test_acc", self.test_acc)
        return loss


print(CNN())


class LoggingCallback(p.Callback):
    def on_train_epoch_end(self, trainer: p.Trainer, pl_module: p.LightningModule):
        epoch = trainer.current_epoch
        logs = trainer.callback_metrics

        loss = logs.get("train_loss")
        accuracy = logs.get("train_acc")

        print(f"Epoch {epoch} - Training Loss: {loss} - Accuracy: {accuracy}")

    def on_validation_epoch_end(
        self, trainer: p.Trainer, pl_module: p.LightningModule
    ) -> None:
        epoch = trainer.current_epoch
        logs = trainer.callback_metrics

        accuracy = logs.get("val_acc")
        loss = logs.get("val_loss")

        print(f"Epoch {epoch} - V Loss: {loss} - Accuracy: {accuracy}")


from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

checkpoint_callback = ModelCheckpoint(
    monitor="val_acc",
    save_top_k=1,
    mode="max",
    dirpath="saved/",
    filename="sample-mnist-{epoch:02d}-{val_loss:.2f}",
)

early_stopping_callback = EarlyStopping(monitor="val_loss", patience=5)


datamodule = MNISTDataModule()
model = CNN()
trainer = p.Trainer(
    callbacks=[LoggingCallback(), checkpoint_callback, early_stopping_callback],
    max_epochs=2,
)
trainer.fit(model, datamodule=datamodule)


trainer.test(model=model, datamodule=datamodule)


test_dataloader = datamodule.test_dataloader()

true = []
pred = []

for batch in test_dataloader:
    x, y = batch
    with torch.no_grad():
        y_pred = model(x)

    _, predicted = torch.max(y_pred, 1)
    #     print(predicted)

    true.extend(y.tolist())
    pred.extend(predicted.tolist())
    break


figure = plt.figure(figsize=(10, 8))
cols, rows = 5, 5
for i in range(1, cols * rows + 1):
    #     sample_idx = torch.randint(len(train_data), size=(1,)).item()
    img = x[i]
    figure.add_subplot(rows, cols, i)
    plt.title(pred[i])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()
