import torchvision

from Trainer import Trainer
import training_utils
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import neptune
import torchvision.transforms as transforms
import torch.nn.functional as F

# Set random seed for reproducibility
torch.manual_seed(42)
import os
from os import path


# Define the SimpleModel
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def save_model(self, dir_to_save, device, distributed):
        if not path.exists(dir_to_save):
            os.makedirs(dir_to_save, mode=0o777)
        self.eval()
        path_to_save = path.join(dir_to_save, "model.pt")
        if not distributed and device.type == 'cuda':
            torch.save(self.state_dict().cpu(), path_to_save)
        else:
            torch.save(self.state_dict(), path_to_save)

    def load_model(self, dir_to_load):
        path_to_load = path.join(dir_to_load, "model.pt")
        state_dict = torch.load(path_to_load)
        self.load_state_dict(state_dict)


if __name__ == '__main__':
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    input_size = 28 * 28  # MNIST image size
    hidden_size = 128
    output_size = 10  # Number of classes in MNIST
    learning_rate = 0.001
    num_epochs = 10
    batch_size = 64

    # Load MNIST dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])

    # Create datasets for training & validation, download if necessary
    training_set = torchvision.datasets.FashionMNIST('./data', train=True,
                                                     transform=transform,
                                                     download=True)
    validation_set = torchvision.datasets.FashionMNIST('./data', train=False,
                                                       transform=transform,
                                                       download=True)

    # Create data loaders for our datasets; shuffle for training, not for validation
    training_loader = torch.utils.data.DataLoader(training_set,
                                                  batch_size=batch_size,
                                                  shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_set,
                                                    batch_size=batch_size,
                                                    shuffle=False)
    # Create model instance
    model = SimpleModel()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    snapshot_path = r'\training\debugging'
    dir_best_model = r'\training\best_debugging'
    nept = neptune.init_run(project="psts-through-unsupervised-learning/psts",
                            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzODRhM2YzNi03Nzk4LTRkZDctOTJiZS1mYjMzY2EzMDMzOTMifQ==")
    train_params = {"train_dataloader": training_loader,
                    "validation_dataloader": validation_loader,
                    "optimizer": optimizer,
                    "loss": loss_fn,
                    "docu_per_batch": 100
                    }

    trainer = Trainer(model, train_params, 10, snapshot_path, dir_best_model,
                      False, training_utils.run_simple_batch, device, nept)
    trainer.train(6, True)
    print("done")
    # torchrun --standalone --nproc_per_node=1 training/trainning_debug_script.py
