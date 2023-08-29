import os
import numpy as np
from dataclasses import dataclass

@dataclass
class TrainConfig():
    exp_name: str = 'test'
    num_classes: int = 10
    num_epochs: int = 10
    batch_size: int = 4
    train_df_path = r"C:\Users\shiri\Documents\School\Master\Research\Context\data\train_set_10_100.csv"

