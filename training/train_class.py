import pickle
from models.models import PstsDecoder
import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torch
import neptune
import torch
import torch.nn.functional as F

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os


class Trainer:
    def __init__(self, model, model_path, train_params, gpu_id, run_docu):
        self.model = model if model else PstsDecoder.load_model(model_path)
        self.epoch_number = train_params['epoch_number'] \
            if 'epoch_number' in train_params else 0
        self.best_v_loss = train_params[
            'best_v_loss'] if 'best_v_loss' in train_params else 1000000
        self.train_batch_size = train_params['train_batch_size']
        self.val_batch_size = train_params['val_batch_size']
        self.train_dataset = train_params['train_dataset']
        self.validation_dataset = train_params['validation_data_set']
        self.loss_function = train_params['loss_function']
        self.optimizer = train_params['optimizer']
        self.model_path = train_params['model_path']  # the last model achieved
        self.model_path_dir = train_params[
            'model_path_dir']  # dir to all models
        self.path_to_save_trainer = train_params['path_to_save_trainer']
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.docu_per_batch = train_params['validation_per_batch'] \
            if 'validation_per_batch' in train_params else 1000
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.run_docu = run_docu
        self.distributed = train_params['distributed']  # True iff multiple GPUs
        self.training_loader = None
        self.validation_loader = None
        self.gpu_id = gpu_id
        if self.distributed:
            self.turn_distributed()
        else:
            self.gpu_id = 0

    def update_timestamp(self):
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    def turn_distributed(self):
        self.model = DDP(self.model, device_ids=[self.gpu_id])

    def set_train_dataloader(self):
        if self.distributed:
            self.training_loader = torch.utils.data.DataLoader(
                self.train_dataset,
                self.train_batch_size,
                shuffle=False,
                sampler=DistributedSampler(
                    self.train_dataset))
        else:
            self.training_loader = torch.utils.data.DataLoader(
                self.train_dataset,
                self.train_batch_size, shuffle=True)

    def set_validation_dataloader(self):
        if self.distributed:
            self.validation_loader = torch.utils.data.DataLoader(
                self.validation_dataset,
                self.val_batch_size,
                shuffle=False,
                sampler=DistributedSampler(
                    self.validation_dataset))
        else:
            self.validation_loader = torch.utils.data.DataLoader(
                self.validation_dataset,
                self.val_batch_size, shuffle=True)

    def save_trainer(self):
        """save the train object in path_to_save_trainer
        the model itself saved separetly in model_path
        datasets and trainloaders (train \validation) won't be saved"""
        if self.gpu_id == 0:
            if self.distributed:
                self.model.module.save_model(self.model_path, self.device,
                                             self.distributed)
            else:
                self.model.save_model(self.model_path, self.device,
                                      self.distributed)
            self.model = None
            self.train_dataset = None
            self.validation_dataset = None
            self.training_loader = None
            self.validation_loader = None
            with open(self.path_to_save_trainer, 'wb') as outp:
                pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load_trainer(cls, path_to_load, train_dataset, validation_dataset):
        with open(path_to_load, 'rb') as inp:
            trainer = pickle.load(inp)
        trainer.model.load_model(trainer.model_path)
        trainer.train_dataset = train_dataset
        trainer.validation_dataset = validation_dataset
        return trainer

    def run_one_batch(self, batch):
        videos, audios, _ = batch
        if self.distributed:
            videos = videos.to(self.gpu_id)
            audios = audios.to(self.gpu_id)
        else:
            videos = videos.to(self.device)
            audios = audios.to(self.device)
        encode_videos, encode_audios = self.model(videos, audios)
        return self.loss_function(encode_videos, encode_audios)

    def train_one_epoch(self, epoch_index):
        self.model.train(True)  # Make sure gradient tracking is on
        running_loss = 0.
        last_loss = 0.
        if not self.training_loader:
            self.set_train_dataloader()
        for i, batch in enumerate(self.training_loader):
            self.optimizer.zero_grad()
            loss = self.run_one_batch(batch)
            loss.backward()  # backpropagation the loss
            self.optimizer.step()  # Adjust learning weights

            running_loss += loss.item()  # Gather data and report
            if i % self.docu_per_batch == self.docu_per_batch - 1:
                last_loss = running_loss / self.docu_per_batch  # loss per batch
                print('batch {} loss: {}'.format(i + 1, last_loss))

                tb_x = epoch_index * len(self.training_loader) + i + 1
                self.run_docu['train/loss avg in last 1000 batches'].append(
                    last_loss)
                self.run_docu['train/loss avg indexes'].append(tb_x)
                running_loss = 0.

        return last_loss

    def run_validation(self, avg_loss, epoch_number):
        self.model.eval()  # Set the model to evaluation mode
        running_vloss = 0.0
        if not self.validation_loader:
            self.set_validation_dataloader()
        with torch.no_grad():  # Disable gradient computation
            for i, vdata in enumerate(self.validation_loader):
                vloss = self.run_one_batch(vdata)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        self.run_docu['training/avg trainning loss per epoch'].append(avg_loss)
        self.run_docu['validation/avg validation loss per epoch'].append(
            avg_vloss)
        self.run_docu['validation/epoch records'].append(epoch_number)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Track the best performance, and save the model's state
        if avg_vloss < self.best_vloss:
            self.best_vloss = avg_vloss
            if self.model_path_dir and self.gpu_id == 0:
                base_name = 'model_{}_{}'.format(self.timestamp, epoch_number)
                path_to_save = os.path.join(self.model_path_dir, base_name)
                self.model_path = path_to_save
                if self.distributed:
                    self.model.module.save_model(path_to_save, self.device,
                                                 self.distributed)
                else:
                    self.model.save_model(path_to_save, self.device,
                                                 self.distributed)

        return self.best_vloss

    def train(self, total_epochs, save_at_end=False):
        self.update_timestamp()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.set_train_dataloader()
        self.set_validation_dataloader()
        for epoch in range(self.epoch_number, total_epochs):
            print('EPOCH {}:'.format(epoch + 1))
            if self.distributed:
                self.training_loader.sampler.set_epoch(epoch)
                self.validation_loader.sampler.set_epoch(epoch)

            avg_loss = self.train_one_epoch(epoch + 1)
            self.best_v_loss = self.run_validation(avg_loss, epoch + 1)

            self.run_docu['validation/best_v_loss'] = self.best_v_loss
            self.run_docu['run_params/epoch_number'] = self.epoch_number = \
                epoch + 1

        self.run_docu.stop()
        if save_at_end:
            self.save_trainer()
        return self.model
