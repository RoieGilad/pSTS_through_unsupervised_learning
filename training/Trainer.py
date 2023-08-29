import os
import time
from datetime import datetime
from os import path
from training.training_utils import run_simple_batch, ddp_setup
import neptune
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

class Trainer:
    """
    Trainer class for training a PyTorch model.
    *This implementation handles runs on both CPU and GPU.
    *To run this class on single\multiple GPUs or to ensure crash consistency,
    use torchrun (from your cmd:
    "torchrun --standalone --nproc_per_node=<num_GPUs> <your_script>").

    *Monitoring is done via neptune.ai.

    The model should have the following functions:
    1) save_model(self, dir_to_save, device, distributed) - to save the model
       in dir_to_save. The function should handle the device and distributed
       manner (only the master will save the model).
    2) load_model(self, dir_to_load) - function to load the model from the
    given directory.

    * the Trainer should be given run_one_batch function with the following
    signature: run_batch(loss, model, batch, distributed:bool , gpu_id:int,
     device:bool)

    Args:
        model (torch.nn.Module): The PyTorch model to be trained.
        train_params (dict): A dictionary containing the training parameters:
            {"batch_size": batch_size,
            "train_dataset": dataset object,
            "validation_dataset": dataset object,
            "optimizer": optimizer object,
            "loss": callable loss function,
            "docu_per_batch": (optional) the batch interval to add monitoring}

        save_every (int): The interval at which to save model snapshots.
        snapshot_path (str): The directory path to save model snapshots.
        dir_best_model (str): The directory path to save the best model.
        distributed (bool): Flag indicating distributed training or GPU training.
        model.
        device (str): The device to run the training on.
        run_docu: neptune.ai monitoring object.
        run_one_batch (callable): Callback function to run a single batch of the
            default= simple batch function

"""

    def __init__(
            self,
            model: torch.nn.Module,
            unused_paramter_idx: list,
            train_params: dict,
            save_every: int,
            snapshot_path: str,
            dir_best_model: str,
            distributed: bool,
            device,
            run_docu,
            run_one_batch=run_simple_batch
    ) -> None:
        if distributed:
            ddp_setup()
        self.gpu_id = int(os.environ["LOCAL_RANK"] if "LOCAL_RANK" in os.environ
                                                      and distributed else 0)
        self.distributed = distributed
        self.device = device
        self.model = model.to(self.gpu_id if self.distributed else self.device)
        self.run_docu = run_docu
        self.dir_best_model = dir_best_model
        self.batch_size = train_params["batch_size"]
        self.train_dataloader = self.set_dataloader(
            train_params["train_dataset"])
        self.validation_dataloader = self.set_dataloader(
            train_params["validation_dataset"])
        self.optimizer = train_params["optimizer"]
        self.loss = train_params["loss"]
        self.best_vloss = train_params[
            'best_vloss'] if 'best_vloss' in train_params else 1000000
        self.docu_per_batch = train_params['docu_per_batch'] \
            if 'docu_per_batch' in train_params else 1000
        self.save_every = save_every
        self.epochs_run = 0
        self._run_batch = run_one_batch
        self.snapshot_path = snapshot_path
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)
            self.model = self.model.to(self.gpu_id if self.distributed else self.device)

        if self.distributed:
            self.model = DDP(self.model, device_ids=[self.gpu_id],
                             #find_unused_parameters=True if unused_paramter_idx,
                             static_graph= True if unused_paramter_idx else False)

    def set_dataloader(self, dataset):
        if self.distributed:
            return torch.utils.data.DataLoader(dataset, self.batch_size,
                                               shuffle=False,
                                               sampler=DistributedSampler(
                                                   dataset))
        else:
            return torch.utils.data.DataLoader(dataset, self.batch_size,
                                               shuffle=True)

    def _load_snapshot(self, snapshot_path):
        path_to_trainer = os.path.join(snapshot_path, "trainer")
        path_to_optimizer = os.path.join(self.snapshot_path, "optimizer")
        if self.distributed:
            loc = f"cuda:{self.gpu_id}"
            snapshot = torch.load(path_to_trainer, map_location=loc)
        else:
            snapshot = torch.load(path_to_trainer)
        optimizer_state = torch.load(path_to_optimizer)
        self.optimizer.load_state_dict(optimizer_state)
        self.model.load_model(snapshot["MODEL_STATE_PATH"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _save_snapshot(self, epoch):
        if not path.exists(self.snapshot_path):
            os.makedirs(self.snapshot_path, mode=0o777)
        path_to_model = os.path.join(self.snapshot_path, "model")
        path_to_trainer = os.path.join(self.snapshot_path, "trainer")
        path_to_optimizer = os.path.join(self.snapshot_path, "optimizer")
        if self.distributed:
            self.model.module.save_model(path_to_model, self.device,
                                         self.distributed)
        else:
            self.model.save_model(path_to_model, self.device,
                                  self.distributed)
        snapshot = {
            "MODEL_STATE_PATH": path_to_model,
            "EPOCHS_RUN": epoch,
            "path_to_optimizer": path_to_optimizer
        }
        torch.save(self.optimizer.state_dict(), path_to_optimizer)
        torch.save(snapshot, path_to_trainer)
        print(
            f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def run_batch(self, batch):
        return self._run_batch(self.loss, self.model, batch,
                               self.distributed, self.gpu_id, self.device)

    def _run_epoch(self, epoch):
        self.model.train(True)  # Make sure gradient tracking is on
        running_loss = 0.
        sample_loss = 0.
        b_sz = len(next(iter(self.train_dataloader))[0])
        print(f"[{'GPU' if self.distributed else 'CPU'}{self.gpu_id}] Epoch "
              f"{epoch} | Batchsize: {b_sz} | Steps: {len(self.train_dataloader)}")
        if self.distributed:
            self.train_dataloader.sampler.set_epoch(epoch)
        for i, batch in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()
            loss = self.run_batch(batch)
            loss.backward()  # backpropagation the loss
            self.optimizer.step()  # Adjust learning weights

            running_loss += loss.item()  # Gather data_processing and report
            sample_loss += loss.item()
            if i % self.docu_per_batch == self.docu_per_batch - 1:
                last_loss = sample_loss / self.docu_per_batch  # loss per batch
                if self.gpu_id == 0:
                    print('batch {} loss: {}'.format(i + 1, last_loss))

                    self.run_docu[f'training/loss avg sample every ' \
                                  f'{self.docu_per_batch} batches'].append(
                        last_loss)
                sample_loss = 0.

        return running_loss / len(self.train_dataloader)

    def _run_validation(self, avg_loss, epoch_number):
        self.model.eval()  # Set the model to evaluation mode
        running_vloss = 0.0
        if self.distributed:
            self.validation_dataloader.sampler.set_epoch(epoch_number)
        with torch.no_grad():  # Disable gradient computation
            for i, vdata in enumerate(self.validation_dataloader):
                running_vloss += self.run_batch(vdata)
        avg_vloss = running_vloss / (i + 1)
        if self.gpu_id == 0:
            self.run_docu['training/avg trainning loss per epoch'].append(
                avg_loss)
            self.run_docu['validation/avg validation loss per epoch'].append(
                avg_vloss)
            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Track the best performance, and save the model's state
        if avg_vloss < self.best_vloss:
            self.best_vloss = avg_vloss
            if self.dir_best_model and self.gpu_id == 0:
                if self.distributed:
                    self.model.module.save_model(self.dir_best_model, self.device,
                                             self.distributed)
                else:
                    self.model.save_model(self.dir_best_model, self.device,
                                          self.distributed)
        return self.best_vloss

    def train(self, max_epochs: int, save_at_end: bool):
        self.model.to(self.gpu_id if self.distributed else self.device)
        self.model.train()
        for epoch in range(self.epochs_run, max_epochs):
            print('EPOCH {}:'.format(epoch + 1))
            time_start = time.time()
            avg_loss = self._run_epoch(epoch + 1)
            self.best_vloss = self._run_validation(avg_loss, epoch + 1)
            print(f'Epoch {epoch+1} took {time.time() - time_start}')
            if self.gpu_id == 0:
                self.run_docu['validation/best_vloss'] = self.best_vloss
                self.run_docu['run_params/epoch_number'] = self.epochs_run = \
                    epoch + 1
            if ((self.distributed and self.gpu_id == 0) or (not self.distributed)) \
                    and epoch % self.save_every == 0:
                self._save_snapshot(epoch + 1)
            self.model.train()

        self.run_docu.stop()
        if save_at_end and ((self.distributed and self.gpu_id == 0) or
                            (not self.distributed)):
            self._save_snapshot(self.epochs_run)
        return self.model




