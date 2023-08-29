import glob
import os
import shutil
import time
from os import path, remove
import random
from data_processing import data_prep as dp
import pandas as pd
from models.models import PstsDecoder
from models import params_utils as pu
import time
import torch
from torchmetrics.functional import pairwise_cosine_similarity
from models.models import PstsDecoder
import data_processing.data_utils as du
from data_processing.dataset_types import VideoDataset, AudioDataset, \
    CombinedDataset
import neptune
from tqdm import tqdm
from models import params_utils as pu
from training.training_utils import run_one_batch_psts
from Loss.pstsLoss import pstsLoss
import evaluate_script as es



def prepare_data_for_preprocessing(src_dir):
    video_dest = "stimuli_video_data"
    audio_dest = "stimuli_audio_data"
    os.makedirs(video_dest, exist_ok=True)
    os.makedirs(audio_dest, exist_ok=True)

    video_samples = sorted(
        glob.glob(path.join(src_dir, '*.mp4')))
    print(video_samples)
    audio_samples = sorted(
        glob.glob(path.join(src_dir, '*.wav')))
    for video_file, audio_file in zip(video_samples, audio_samples):
        video_file_name = video_file.split("\\")[1]
        audio_file_name = audio_file.split("\\")[1]
        id = "id_" + video_file_name.split("_")[0]
        sample_number = video_file_name.split("_")[1].split(".")[0]
        id_video_dir = os.path.join(video_dest, id)
        id_audio_dir = os.path.join(audio_dest, id)

        os.makedirs(id_video_dir, exist_ok=True)
        os.makedirs(id_audio_dir, exist_ok=True)

        # Create the sample directory inside the ID directory
        video_num_dir = os.path.join(id_video_dir, f"video_{sample_number}")
        audio_num_dir = os.path.join(id_audio_dir, f"audio_{sample_number}")
        print(video_num_dir)

        os.makedirs(video_num_dir, exist_ok=True)
        os.makedirs(audio_num_dir, exist_ok=True)

        # Move the file to the sample directory
        shutil.move(os.path.join(src_dir, video_file_name), os.path.join(video_num_dir, video_file_name))
        shutil.move(os.path.join(src_dir, audio_file_name), os.path.join(audio_num_dir, audio_file_name))

def create_vox_samples_dir_for_rsa(test_dir):
    file_path = r"dataset/test/160k_test_500ms/data_md_0.xlsx"
    df = pd.read_excel(file_path)
    destination_directory1 = 'vox_samples_rsa_60'
    destination_directory2 = 'vox_samples_rsa_30'
    small_list = []
    large_list = []

    num_samples_to_copy = 30

    # List all directories in the source directory
    all_ids = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]

    # Randomly select num_samples_to_copy directories
    selected_ids = random.sample(all_ids, num_samples_to_copy)

    # Create the destination directory if it doesn't exist
    os.makedirs(destination_directory1, exist_ok=True)
    os.makedirs(destination_directory2, exist_ok=True)

    # Copy the selected directories and their contents to the destination directory
    for speaker_id in selected_ids:
        speaker_data = df[df['speaker_id'] == speaker_id]
        # Get unique 'num_of_video' values for the current speaker
        unique_num_values = speaker_data['id_part'].unique()

        # If there are less than 2 unique 'num_of_video' values, skip this speaker
        if len(unique_num_values) < 2:
            print("there is a speaker with one video")
            continue

        # Get the first two unique 'num_of_video' values
        num_value_1 = unique_num_values[0]
        num_value_2 = unique_num_values[1]

        # Get the sample indices corresponding to these unique 'num_of_video' values
        sample_index_1 = speaker_data[speaker_data['id_part'] == num_value_1].iloc[0]['sample_index']
        sample_index_2 = speaker_data[speaker_data['id_part'] == num_value_2].iloc[0]['sample_index']
        small_list.append(sample_index_1)
        large_list.append(sample_index_1)
        large_list.append(sample_index_2)
        dir_name1 = f"sample_{sample_index_1}"
        dir_name2 = f"sample_{sample_index_2}"
        source_path1 = os.path.join(r"dataset/test/160k_test_500ms", dir_name1)
        source_path2 = os.path.join(r"dataset/test/160k_test_500ms", dir_name2)

        destination_path1 = os.path.join(destination_directory1, dir_name1)
        destination_path2 = os.path.join(destination_directory1, dir_name2)
        destination_path3 = os.path.join(destination_directory2, dir_name1)

        shutil.copytree(source_path1, destination_path1)
        shutil.copytree(source_path2, destination_path2)
        shutil.copytree(source_path1, destination_path3)

        print(f"Copying {dir_name1} and {dir_name2} of speaker {speaker_id} to destination.")

    print("Copying completed.")
    filtered_data_30 = df[df["sample_index"].isin(small_list)]
    filtered_data_60 = df[df["sample_index"].isin(large_list)]
    filtered_file_path_30 = "vox_samples_rsa_30/data_md.xlsx"
    filtered_data_30.to_excel(filtered_file_path_30, index=False)
    filtered_file_path_60 = "vox_samples_rsa_60/data_md.xlsx"
    filtered_data_60.to_excel(filtered_file_path_60, index=False)

    dp.get_mean_and_std(destination_directory1)
    dp.get_mean_and_std(destination_directory2)

def get_model(path_to_load, num_frames=3):
    batch_size = 30
    use_end_frame = True
    use_decoder = True

    model_params = {'batch_size': batch_size,
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
    psts_encoder = PstsDecoder(psts_params, False, use_end_frame, use_decoder)
    psts_encoder.load_model(path_to_load)
    return psts_encoder

def get_psts_representation(model, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    num_of_samples = len(dataset)
    outputs = [[None, None, None] for _ in range(num_of_samples)]

    for i in tqdm(range(num_of_samples)):
        v1, a1, l1 = dataset[i]
        encode_v1, encode_a1 = es.fast_run_and_compare(model, v1, a1, device)
        outputs[i] = [encode_v1, encode_a1, l1]
    return outputs


if __name__ == '__main__':
    #prepare_data_for_preprocessing("stimuli")
    #create_vox_samples_dir_for_rsa("dataset/test/voxceleb_video_test/mp4")

    seed = 42
    torch.manual_seed(seed)
    data_dir = r'vox_samples_rsa_30'
    best_model_dir = r'models/check transformer whole DS, no gradient BS= 54, num frames=3, end_frame=True, LR= 0.0000001, drop=0.3, dim_feedforward=2048, num_outputfeature=512, train=0.9, num_heads=4, num_layers=2/best_model'
    dataset = es.get_dataset(data_dir)
    model = get_model(best_model_dir)
    #neptune = neptune.init_run(
     #   project="psts-through-unsupervised-learning/psts",
      #  api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzODRhM2YzNi03Nzk4LTRkZDctOTJiZS1mYjMzY2EzMDMzOTMifQ==")
    print(get_psts_representation(model, dataset))