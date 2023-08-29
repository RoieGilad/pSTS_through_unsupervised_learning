import os
import numpy as np
from dataclasses import dataclass

@dataclass
class TrainSetConfig():
    num_classes: int = 10
    train_num_instances_per_class: int = 100
    val_num_instances_per_class: int = 30

    data_dir: str = r"C:\Users\shiri\Documents\School\Master\Research\Face_recognition\Data\VGG-Face2\MTCNN\train_mtcnn_aligned_2000"
    root_dir: str = r"C:\Users\shiri\Documents\School\Master\Research\Context"