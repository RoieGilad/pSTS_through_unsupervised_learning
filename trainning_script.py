import torchvision
import math
import Loss
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
from data_processing.dataset_types import VideoDataset, AudioDataset, \
    CombinedDataset
import data_processing.data_utils as du
from models.models import VideoDecoder, AudioDecoder, PstsDecoder
from models import params_utils as pu
from torch.utils.data import DataLoader, random_split

# Set random seed for reproducibility
torch.manual_seed(42)
import os
from os import path
from Loss.pstsLoss import pstsLoss
import training.training_utils as tu

data_dir = os.path.join("demo_data", "demo_after_flattening")
transforms_dict = {'a_frame_transform': du.train_a_frame_transformer,
                   'a_batch_transform': du.train_audio_transformer,
                   'v_frame_transform': du.train_v_frame_transformer,
                   'v_batch_transform': du.train_video_transformer}


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


def split_dataset(dataset, ratio=0.9):
    size1 = int(ratio * len(dataset))
    size2 = len(dataset) - size1
    # size1 = 20480
    # size2 = 2048
    size3 = len(dataset) - size1 - size2
    return random_split(dataset, [size1, size2, size3])


def run_train(model, train_dataset, validation_dataset, batch_size, run_id,
              nept, snapshot_path, dir_best_model, unused_parameters):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learning_rate = 0.0000001

    train_params = {"train_dataset": train_dataset,
                    "validation_dataset": validation_dataset,
                    "optimizer": optim.Adam(model.parameters(),
                                            lr=learning_rate),
                    "loss": pstsLoss(),
                    "batch_size": batch_size,
                    "docu_per_batch": 1,
                    "learning_rate": learning_rate,
                    "optimizer_name": "Adam",
                    "loss_name": "pstsLoss",
                    "snapshot_path": snapshot_path,
                    'dir_best_model': dir_best_model,
                    'model_run_id': run_id
                    }
    nept['params/train_params'] = train_params

    trainer = Trainer(model, unused_parameters, train_params, 100,
                      snapshot_path, dir_best_model,
                      True, device, nept, tu.run_one_batch_psts)
    trainer.train(40, True)  # todo change the maximal epoch to reach
    print("done")
    # torchrun --standalone --nproc_per_node=2 pSTS_through_unsupervised_learning/trainning_script.py


def prepare_model_dataset_and_run(run_id, snapshot_path, dir_best_model):
    nept = neptune.init_run(project="psts-through-unsupervised-learning/psts",
                            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzODRhM2YzNi03Nzk4LTRkZDctOTJiZS1mYjMzY2EzMDMzOTMifQ==")
    seed = 42
    dataset_dir = os.path.join(r'dataset', "160k_train_500ms")
    batch_size = 54
    num_frames = 3  # number of none ending frames (sequance will be +int(use_end_frame))
    use_end_frame = True
    use_decoder = True
    torch.manual_seed(seed)

    print(
        f'expected uniform probability loss: {2 * math.log(batch_size * (num_frames + int(use_end_frame)))}')
    unused_parameters = []

    model_params = {'dataset_dir': dataset_dir,
                    'batch_size': batch_size,
                    'num_frames': num_frames,
                    'use_end_frame': use_end_frame,
                    'use_decoder': use_decoder,
                    'dim_resnet_to_transformer': 2048,
                    'num_heads': 4,
                    'num_layers': 2,
                    'batch_first': True,
                    'dim_feedforward': 2048,
                    # equal to dim_resnet_to_transformer
                    'num_output_features': 512,
                    'dropout': 0.3,
                    'mask': torch.triu(
                        torch.ones(num_frames + int(use_end_frame),
                                   num_frames + int(use_end_frame)), 1).bool(),
                    'seed': seed}
    nept['params/model_params'] = model_params

    combined_dataset = CombinedDataset(model_params["dataset_dir"],
                                       du.get_label_path(
                                           model_params["dataset_dir"]),
                                       transforms_dict,
                                       num_frames=num_frames,
                                       test=False,
                                       step_size=1)

    video_params = pu.init_Video_decoder_params(num_frames=num_frames,
                                                dim_resnet_to_transformer=
                                                model_params[
                                                    "dim_resnet_to_transformer"],
                                                num_heads=model_params[
                                                    "num_heads"],
                                                dim_feedforward=model_params[
                                                    "dim_feedforward"],
                                                batch_first=model_params[
                                                    "batch_first"],
                                                num_layers=model_params[
                                                    "num_layers"],
                                                num_output_features=
                                                model_params[
                                                    "num_output_features"],
                                                mask=model_params["mask"],
                                                dropout=model_params["dropout"],
                                                max_len=100,
                                                use_decoder=use_decoder,
                                                use_end_frame=use_end_frame)
    audio_params = pu.init_audio_decoder_params(num_frames=num_frames,
                                                dim_resnet_to_transformer=
                                                model_params[
                                                    "dim_resnet_to_transformer"],
                                                num_heads=model_params[
                                                    "num_heads"],
                                                dim_feedforward=model_params[
                                                    "dim_feedforward"],
                                                batch_first=model_params[
                                                    "batch_first"],
                                                num_layers=model_params[
                                                    "num_layers"],
                                                num_output_features=
                                                model_params[
                                                    "num_output_features"],
                                                mask=model_params["mask"],
                                                dropout=model_params["dropout"],
                                                max_len=100,
                                                use_decoder=use_decoder,
                                                use_end_frame=use_end_frame)

    psts_params = pu.init_psts_decoder_params(num_frames=num_frames,
                                              video_params=video_params,
                                              audio_params=audio_params)

    psts_encoder = PstsDecoder(psts_params, True, use_end_frame, use_decoder)
    psts_encoder.load_resnet(
        r'models/continue ---models/check linear transforms LR = 0.0001 whole DS, rnadom frame & interval, train - 0.9. 4 frames/best_model')  # load previous trainning of resnet
    psts_encoder.set_resnet_gradient(False, False)

    train_combined_dataset, validation_combined_dataset, _ = split_dataset(
        combined_dataset)  # split to validation

    run_train(psts_encoder, train_combined_dataset, validation_combined_dataset,
              batch_size, run_id, nept, snapshot_path, dir_best_model,
              unused_parameters)


if __name__ == '__main__':
    # run_id = "sanity check resnet 20k, 1 frame"
    run_id = "check transformer whole DS, no gradient BS= 54, num frames=3, end_frame=True, LR= 0.0000001, drop=0.3, dim_feedforward=2048, num_outputfeature=512, train=0.9, num_heads=4, num_layers=2"
    snapshot_path = os.path.join("models", str(run_id), "snapshot")
    dir_best_model = os.path.join("models", str(run_id), "best_model")
    print(f'the current run_id to be run: {run_id}')
    prepare_model_dataset_and_run(run_id, snapshot_path, dir_best_model)
