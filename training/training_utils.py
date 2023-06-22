import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import torch
from train_class import Trainer


def ddp_setup(rank: int, world_size: int, master_addr="localhost",
              master_port="12355"):
    """
    Args:
        rank: Unique identifier of each process
       world_size: Total number of processes
       master_addr: the IP of the machine that's run the rank 0 process
       master_port: free ports in the master machine (12355 all free ports)

       **master = the machine to synchronize all the other machines
    """
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    # nccl for nvidia gpu's
    torch.cuda.set_device(rank)


def main_distributed(train_params, trainer_path, model, model_path, run_docu,
                     total_epochs, train_ds=None, validation_ds=None,
                     save_at_end=False):
    """trains_params if given should contain Dataset, otherwise
    datasets should be given separately"""
    world_size = torch.cuda.device_count()
    mp.spawn(train_distributed,
             args=(world_size, train_params, trainer_path, model,
                   model_path, run_docu, total_epochs,
                   train_ds, validation_ds, save_at_end),
             nprocs=world_size)


def train_distributed(rank, world_size, train_params, trainer_path, model,
                      model_path, run_docu, total_epochs, train_ds=None,
                      validation_ds=None, save_at_end=False):
    """trains_params if given should contain Dataset, otherwise
    datasets should be given separately"""
    ddp_setup(rank, world_size)
    trainer = Trainer(model, model_path, train_params, rank,
                      run_docu) if train_params else Trainer.load_trainer(
        trainer_path, train_ds, validation_ds)
    trainer.train(total_epochs, save_at_end)
    destroy_process_group()


def main_single_machine(train_params, trainer_path, model, model_path, run_docu,
                        total_epochs, train_ds=None, validation_ds=None,
                        save_at_end=False):
    """trains_params if given should contain Dataset, otherwise
    datasets should be given separately"""
    trainer = Trainer(model, model_path, train_params, 0,
                      run_docu) if train_params else Trainer.load_trainer(
        trainer_path, train_ds, validation_ds)
    trainer.train(total_epochs, save_at_end)
