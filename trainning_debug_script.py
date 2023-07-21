import torchvision

from training.Trainer import Trainer
from training import training_utils
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import neptune
import torchvision.transforms as transforms
import torch.nn.functional as F
from data_processing.dataset_types import VideoDataset
import data_processing.data_utils as du
from models.models import VideoDecoder, AudioDecoder, PstsDecoder
from models import params_utils as pu
from torch.utils.data import DataLoader
# Set random seed for reproducibility
torch.manual_seed(42)
import os
from os import path

data_dir = os.path.join("demo_data", "demo_after_flattening")


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


def run_train(model, train_dataset, validation_dataset, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learning_rate = 0.001

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    snapshot_path = os.path.join("debugging", "snapshot")
    dir_best_model = os.path.join("debugging", "best_model")
    nept = neptune.init_run(project="psts-through-unsupervised-learning/psts",
                            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzODRhM2YzNi03Nzk4LTRkZDctOTJiZS1mYjMzY2EzMDMzOTMifQ==")
    train_params = {"train_dataset": train_dataset,
                    "validation_dataset": validation_dataset,
                    "optimizer": optimizer,
                    "loss": loss_fn,
                    "batch_size": batch_size,
                    "docu_per_batch": 5
                    }

    trainer = Trainer(model, train_params, 10, snapshot_path, dir_best_model,
                      False, device, nept)
    trainer.train(2, True)
    print("done")
    # torchrun --standalone --nproc_per_node=1 training/trainning_debug_script.py


def main():
    num_frames = 16
    dim_resnet_to_transformer = 1024
    num_heads = 4
    num_layers = 4
    batch_first = True
    dim_feedforward = dim_resnet_to_transformer
    num_output_features = 512
    dropout = 0.1
    mask = torch.triu(torch.ones(num_frames, num_frames), 1).bool()

    batch_size = 16

    video_dataset = VideoDataset(data_dir, du.get_label_path(data_dir),
                               du.train_v_frame_transformer,
                               du.train_end_v_frame_transformer,
                               du.train_video_transformer,
                               num_frames=num_frames,
                               test=False,
                               step_size=1)
    video_loader = DataLoader(video_dataset, batch_size=batch_size, shuffle=True)

    video_params = pu.init_Video_decoder_params(num_frames=num_frames,
                                                dim_resnet_to_transformer=1024,
                                                num_heads=num_heads,
                                                dim_feedforward=dim_feedforward,
                                                batch_first=batch_first,
                                                num_layers=num_layers,
                                                num_output_features=num_output_features,
                                                mask=mask,
                                                dropout=dropout, max_len=100)
    video_decoder = VideoDecoder(video_params, True)

    run_train(video_decoder, video_dataset, video_dataset, batch_size)


if __name__ == '__main__':
    main()
