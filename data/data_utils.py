import glob
from os import path

import torchvision.transforms as transforms

train_v_frame_transformer = transforms.Compose([
    transforms.Resize((256, 256)), transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

train_video_transformer = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(),
    transforms.RandomCrop([224, 224])])

train_a_frame_transformer = transforms.Compose([   # TODO think about what we do here, which size, how to normalize, add noise and how toTensor
    transforms.Resize((256, 256)), transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

train_audio_transformer = transforms.Compose([  # TODO same
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(),
    transforms.RandomCrop([224, 224])])


def add_addition_to_path(path, addition):
    """ for a given path = dirname/base_name.type
    will return the new path: dirname/base_name_addition.type"""
    dirname, base_name = path.split(path)
    file_name, file_ext = path.splitext(base_name)
    base_name = "".join([file_name, "_", addition, file_ext])
    return path.join(dirname, base_name)


def folder_iterator_by_path(root_dir: str):
    """yield the folder path of the next sample, in order"""
    for p in sorted(glob.glob(root_dir, recursive=False)):
        if not path.isfile(p):
            yield p

def file_iterator_by_path(root_dir: str):
    """yield the file path of the next sample, in order"""
    for p in sorted(glob.glob(root_dir, recursive=False)):
        if path.isfile(p):
            yield p


def audio_folder_iterator(root_dir: str):
    """return all the file paths matching the following pattern
    samples_root_dir/*/audio, in-order"""
    for p in folder_iterator_by_path(path.join(root_dir, "*", "audio")):
        yield p


def video_folder_iterator(root_dir: str):
    """return all the file paths matching the following pattern
    samples_root_dir/*/video, in-order"""
    for p in folder_iterator_by_path(path.join(root_dir, "*", "video")):
        yield p


def file_iterator_by_type(root_dir: str, type: str):
    """return all path of the files in the root_dir from the type is given
     as input, in-order"""
    for p in sorted(glob.glob(path.join(root_dir, "*." + type))):
        yield p

def get_sample_index(path: str):
    for dir_name in path.split("/"):
        if dir_name.startswith("sample"):
            return dir_name
    return ""


def get_video_frame_rate(data_md, sample_id: str):
    """return the frame rate of sample_id from the given DF"""
    if not sample_id.startswith("sample"):
        return 0
    id = int(sample_id.split("_")[-1])
    return int(data_md.iloc[id, 2])


