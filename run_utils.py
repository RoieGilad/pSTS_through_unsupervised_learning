from torch.distributed import init_process_group, destroy_process_group
from training.training_utils import run_one_batch_psts, ddp_setup
from Loss.pstsLoss import pstsLoss
from functools import *
from training.Trainer import Trainer


def train_distributed(train_params, model, save_every, snapshot_path,
                      dir_best_model, run_docu,
                      total_epochs, train_ds=None, validation_ds=None,
                      save_at_end=False):
    """trains_params if given should contain Dataset, otherwise
    datasets should be given separately"""
    ddp_setup()
    train(train_params, model, save_every, snapshot_path, dir_best_model,
          run_docu,
          total_epochs, train_ds, validation_ds, save_at_end, True)
    destroy_process_group()


def train(train_params, model, save_every, snapshot_path, dir_best_model,
          run_docu,
          total_epochs, train_ds, validation_ds, save_at_end, distributed):
    run_one_batch = partial(run_one_batch_psts, loss=pstsLoss())


    train_params['train_dataset'] = train_ds
    train_params['validation_dataset'] = validation_ds
    trainer = Trainer(model, train_params, save_every, snapshot_path,
                      dir_best_model, distributed, run_one_batch, run_docu)
    trainer.train(total_epochs, save_at_end)


def main_single_machine(train_params, model, save_every, snapshot_path,
                        dir_best_model, run_docu, total_epochs, train_ds=None,
                        validation_ds=None, save_at_end=False):
    """trains_params if given should contain Dataset, otherwise
    datasets should be given separately"""
    train(train_params, model, save_every, snapshot_path, dir_best_model,
          run_docu,
          total_epochs, train_ds, validation_ds, save_at_end, False)


# if __name__ == "__main__":
#     import argparse
#
#     parser = argparse.ArgumentParser(
#         description='simple distributed training job')
#     parser.add_argument('total_epochs', type=int,
#                         help='Total epochs to train the model')
#     parser.add_argument('save_every', type=int,
#                         help='How often to save a snapshot')
#     parser.add_argument('--batch_size', default=32, type=int,
#                         help='Input batch size on each device (default: 32)')
#     args = parser.parse_args()
#
#     # main(args.save_every, args.total_epochs, args.batch_size)