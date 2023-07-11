import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import torch


def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def run_one_batch_psts(loss, model, batch, distributed, gpu_id, device):
    videos, audios, _ = batch
    if distributed:
        videos = videos.to(gpu_id)
        audios = audios.to(gpu_id)
    else:
        videos = videos.to(device)
        audios = audios.to(device)
    encode_videos, encode_audios = model(videos, audios)
    return loss(encode_videos, encode_audios)


def run_simple_batch(loss, model, batch, distributed, gpu_id, device):
    inputs, labels = batch
    if distributed:
        inputs = inputs.to(gpu_id)
        labels = labels.to(gpu_id)

    else:
        inputs = inputs.to(device)
        labels = labels.to(device)
    inputs = model(inputs)
    return loss(inputs, labels)

