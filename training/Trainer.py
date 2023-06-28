import os
from datetime import datetime
import torch
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import neptune
from os import path

def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            train_params: dict,
            save_every: int,
            snapshot_path: str,
            dir_best_model: str,
            distributed: bool,
            run_one_batch,
            device,
            run_docu
    ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"] if "LOCAL_RANK" in os.environ
                          else 0)
        self.distributed = distributed
        self.device = device
        self.model = model.to(self.gpu_id if self.distributed else self.device)
        self.run_docu = run_docu
        self.dir_best_model = dir_best_model
        self.train_dataloader = train_params["train_dataloader"]
        self.validation_dataloader = train_params["validation_dataloader"]
        self.optimizer = train_params["optimizer"]
        self.loss = train_params["loss"]
        self.best_vloss = train_params[
            'best_vloss'] if 'best_v_loss' in train_params else 1000000
        self.docu_per_batch = train_params['docu_per_batch'] \
            if 'docu_per_batch' in train_params else 1000
        self.save_every = save_every
        self.epochs_run = 0
        self._run_batch = run_one_batch
        self.snapshot_path = snapshot_path
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)
        if self.distributed:
            self.model = DDP(self.model, device_ids=[self.gpu_id])

    def _load_snapshot(self, snapshot_path):
        path_to_trainer = os.path.join(snapshot_path, "trainer")
        if self.distributed:
            loc = f"cuda:{self.gpu_id}"
            snapshot = torch.load(path_to_trainer, map_location=loc)
        else:
            snapshot = torch.load(path_to_trainer)
        print(snapshot)
        self.model.load_model(snapshot["MODEL_STATE_PATH"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _save_snapshot(self, epoch):
        if not path.exists(self.snapshot_path):
            os.makedirs(self.snapshot_path, mode=0o777)
        path_to_model = os.path.join(self.snapshot_path, "model")
        path_to_trainer = os.path.join(self.snapshot_path, "trainer")
        if self.distributed:
            self.model.module.save_model(path_to_model, self.device,
                                         self.distributed)
        else:
            self.model.save_model(path_to_model, self.device,
                                  self.distributed)
        snapshot = {
            "MODEL_STATE_PATH": path_to_model,
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, path_to_trainer)
        print(
            f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def run_batch(self, batch):
        return self._run_batch(self.loss, self.model, batch,
                               self.distributed, self.gpu_id, self.device)

    def _run_epoch(self, epoch):
        self.model.train(True)  # Make sure gradient tracking is on
        running_loss = 0.
        last_loss = 0.
        b_sz = len(next(iter(self.train_dataloader))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: "
              f"{len(self.train_dataloader)}")
        if self.distributed:
            self.train_dataloader.sampler.set_epoch(epoch)
        for i, batch in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()
            loss = self.run_batch(batch)
            loss.backward()  # backpropagation the loss
            self.optimizer.step()  # Adjust learning weights

            running_loss += loss.item()  # Gather data and report
            if i % self.docu_per_batch == self.docu_per_batch - 1:
                last_loss = running_loss / self.docu_per_batch  # loss per batch
                print('batch {} loss: {}'.format(i + 1, last_loss))

                tb_x = epoch * len(self.train_dataloader) + i + 1
                self.run_docu['train/loss avg in last 1000 batches'].append(
                    last_loss)
                self.run_docu['train/loss avg indexes'].append(tb_x)
                running_loss = 0.

        return last_loss

    def _run_validation(self, avg_loss, epoch_number):
        self.model.eval()  # Set the model to evaluation mode
        running_vloss = 0.0
        if self.distributed:
            self.validation_dataloader.sampler.set_epoch(epoch_number)
        with torch.no_grad():  # Disable gradient computation
            for i, vdata in enumerate(self.validation_dataloader):
                running_vloss += self.run_batch(vdata)
        avg_vloss = running_vloss / (i + 1)
        self.run_docu['training/avg trainning loss per epoch'].append(avg_loss)
        self.run_docu['validation/avg validation loss per epoch'].append(
            avg_vloss)
        self.run_docu['validation/epoch records'].append(epoch_number)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Track the best performance, and save the model's state
        if avg_vloss < self.best_vloss:
            self.best_vloss = avg_vloss
            if self.dir_best_model and self.gpu_id == 0:
                base_name = 'model_{}_{}'.format(self.timestamp, epoch_number)
                path_to_save = os.path.join(self.dir_best_model, base_name)
                if self.distributed:
                    self.model.module.save_model(path_to_save, self.device,
                                                self.distributed)
                else:
                    self.model.save_model(path_to_save, self.device,
                                                self.distributed)

        return self.best_vloss

    def train(self, max_epochs: int, save_at_end: bool):
        self.model.to(self.gpu_id if self.distributed else self.device)
        self.model.train()
        for epoch in range(self.epochs_run, max_epochs):
            print('EPOCH {}:'.format(epoch + 1))
            avg_loss = self._run_epoch(epoch + 1)
            self.best_v_loss = self._run_validation(avg_loss, epoch + 1)
            self.model.train()
            self.run_docu['validation/best_v_loss'] = self.best_v_loss
            self.run_docu['run_params/epoch_number'] = self.epochs_run = \
                epoch + 1
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch + 1)
                self.model.train()

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name, param.data)
        self.run_docu.stop()
        if save_at_end and self.gpu_id == 0:
            self._save_snapshot(self.epochs_run)
        return self.model