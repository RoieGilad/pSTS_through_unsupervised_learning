import glob
import math
import os
import shutil
import time
from os import path, remove
import random
import matplotlib.pyplot as plt

import numpy as np
from natsort import natsorted
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
import evaluate_script as es
import torchaudio
from speechbrain.pretrained import EncoderClassifier
import pingouin as pg


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

def get_face_model_embeddings(src_dir):
    pass

def get_audio_model_embedding(audio_model, audio_sample):
    """
    audio_sample: a wav file load by torchaudio.load (take the first argument)
    """
    return audio_model.encode_batch(audio_sample)

def get_audio_model_representations(src_dir):
    xlsx_path = path.join(src_dir, "data_md.xlsx")
    md_df = pd.read_excel(xlsx_path)
    representations = []
    speaker_verification_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb")

    samples_directories = natsorted(glob.glob(path.join(src_dir, "sample*")))
    for sample_directory in samples_directories:
        sample_representations = []
        sample_wav_files = natsorted(glob.glob(path.join(sample_directory, 'audio', '*.wav')))
        wav_files_for_embedding = sample_wav_files[1:4]
        sample_index = int(sample_directory.split("sample_")[-1])
        row = md_df[md_df['sample_index'] == sample_index]
        speaker_id = row.iloc[0]['speaker_id']
        for wav_file in wav_files_for_embedding:
            sample_representations.append(get_audio_model_embedding(speaker_verification_model,
                                                                    torchaudio.load(wav_file)[0]))
        # End frame
        sample_representations.append(sum(sample_representations)/3)
        sample_representations.append(speaker_id)
        representations.append(sample_representations)
    return representations

def create_and_save_rdm(embeddings, labels, dest_path):
    rdm = np.zeros((len(embeddings), len(embeddings)))
    for i, first_em in enumerate(embeddings):
        for j, second_em in enumerate(embeddings):
            rdm[i, j] = pairwise_cosine_similarity(first_em, second_em)

    n = len(labels)
    rdm_df = pd.DataFrame(data=rdm, columns=labels, index=labels)
    rdm_df.to_excel(dest_path)

    print("rdm saved successfuly")

def create_audio_model_rdms(save_dir, audio_model_dir):
    # Create whole audio (audio model) rdm
    whole_audio_embeddings = []
    whole_labels = []
    audio_frames_embeddings = []
    audio_frames_labels = []
    save_whole_audio_rdm_path = path.join(save_dir, "audio_model_whole_audio_rdm.xlsx")
    save_audio_frames_rdm_path = path.join(save_dir, "audio_model_audio_frames_rdm.xlsx")
    audio_model_representations = get_audio_model_representations(audio_model_dir)
    for representation in audio_model_representations:
        whole_audio_embeddings.append(representation[-2][0])
        audio_frames_embeddings.append(representation[0][0])
        audio_frames_embeddings.append(representation[1][0])
        audio_frames_embeddings.append(representation[2][0])
        whole_labels.append(representation[-1])
        audio_frames_labels.extend([representation[-1]] * 3)
    print("creating whole audio - audio model - rdm")
    create_and_save_rdm(whole_audio_embeddings, whole_labels, save_whole_audio_rdm_path)
    print("creating audio frames - audio model - rdm")
    create_and_save_rdm(audio_frames_embeddings, audio_frames_labels, save_audio_frames_rdm_path)


def create_psts_rdms(model, data_dir, save_dir):
    dataset = es.get_dataset(data_dir)
    representations = get_psts_representation(model, dataset)
    video_end_frames_embeddings = []
    audio_end_frames_embeddings = []
    end_frames_labels = []
    audio_frames_embeddings = []
    video_frames_embeddings = []
    frames_labels = []
    save_whole_audio_rdm_path = path.join(save_dir, "psts_model_whole_audio_rdm.xlsx")
    save_audio_frames_rdm_path = path.join(save_dir, "psts_model_audio_frames_rdm.xlsx")
    save_whole_video_rdm_path = path.join(save_dir, "psts_model_whole_video_rdm.xlsx")
    save_video_frames_rdm_path = path.join(save_dir, "psts_model_video_frames_rdm.xlsx")
    for representation in representations:
        video_frames_embeddings.append(representation[0][0][0].unsqueeze(0))
        video_frames_embeddings.append(representation[0][0][1].unsqueeze(0))
        video_frames_embeddings.append(representation[0][0][2].unsqueeze(0))
        video_end_frames_embeddings.append(representation[0][0][3].unsqueeze(0))
        audio_frames_embeddings.append(representation[1][0][0].unsqueeze(0))
        audio_frames_embeddings.append(representation[1][0][1].unsqueeze(0))
        audio_frames_embeddings.append(representation[1][0][2].unsqueeze(0))
        audio_end_frames_embeddings.append(representation[1][0][3].unsqueeze(0))
        end_frames_labels.append(representation[-1])
        frames_labels.extend([representation[-1]] * 3)
    print("creating whole audio psts rdm")
    create_and_save_rdm(audio_end_frames_embeddings, end_frames_labels, save_whole_audio_rdm_path)
    print("creating audio frames psts rdm")
    create_and_save_rdm(audio_frames_embeddings, frames_labels, save_audio_frames_rdm_path)
    print("creating whole video psts rdm")
    create_and_save_rdm(video_end_frames_embeddings, end_frames_labels, save_whole_video_rdm_path)
    print("creating video frames psts rdm")
    create_and_save_rdm(video_frames_embeddings, frames_labels, save_video_frames_rdm_path)

def create_rsa_from_two_rdms(path_to_first_rdm, first_rdm_type, path_to_second_rdm,
                             second_rdm_type, dir_to_save_rsa):
    rdm1 = pd.read_excel(path_to_first_rdm, index_col=0).values
    rdm2 = pd.read_excel(path_to_second_rdm, index_col=0).values
    labels = list(pd.read_excel(path_to_first_rdm, index_col=0).index)
    full_corr_file_name = f"{first_rdm_type}_{second_rdm_type}_full_corr_rsa.xlsx"
    partial_corr_file_name = f"{first_rdm_type}_{second_rdm_type}_partial_corr_rsa.xlsx"
    n = rdm1.shape[0]
    indices = np.tril_indices(n, k=-1)  # Exclude main diagonal
    lower_triangle_first_rdm = rdm1[indices]
    lower_triangle_second_rdm = rdm2[indices]

    #full_correlation_matrix = np.corrcoef(rdm1, rdm2)
    full_correlation_df = pd.DataFrame({'Vector1': lower_triangle_first_rdm, 'Vector2': lower_triangle_second_rdm})
    correlation_results = pg.pairwise_corr(full_correlation_df, columns=labels, method='pearson')
    print(correlation_results)
    #full_correlation_df = pd.DataFrame(full_correlation_matrix, index=labels,
       #                                columns=labels)
    #full_correlation_matrix = full_correlation_result['r'].values
    #print(full_correlation_matrix)

    #partial_correlation_result = pg.partial_corr(data=pd.DataFrame({'first_rdm': rdm1.flatten(),
     #                                                        'second_rdm': rdm2.flatten()}),
      #                                    x='first_rdm', y='second_rdm', method='pearson')
    #partial_correlation_matrix = partial_correlation_result['r'].values

    #full_corr_df = pd.DataFrame(full_correlation_result, index=labels, columns=labels)
    #full_correlation_df.to_excel(path.join(dir_to_save_rsa, full_corr_file_name))
    #print(f"saved {full_corr_file_name}")
    #partial_corr_df = pd.DataFrame(partial_correlation_result, index=labels, columns=labels)
    #partial_corr_df.to_excel(path.join(dir_to_save_rsa, partial_corr_file_name))
    #print(f"saved {partial_corr_file_name}")

def plot_rsa(rsa_path):
    correlation_df = pd.read_excel(rsa_path, index_col=0)
    # Create a heatmap of the correlation matrix
    plt.figure(figsize=(8, 6))
    plt.title("Correlation Matrix")
    plt.imshow(correlation_df.values, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar()
    # Set tick labels
    plt.xticks(range(len(correlation_df.columns)), correlation_df.columns, rotation='vertical')
    plt.yticks(range(len(correlation_df.index)), correlation_df.index)
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    #prepare_data_for_preprocessing("stimuli")
    #create_vox_samples_dir_for_rsa("dataset/test/voxceleb_video_test/mp4")

    seed = 42
    torch.manual_seed(seed)
    data_dir = r'vox_samples_rsa_30'
    best_model_dir = r'models/check transformer whole DS, no gradient BS= 54, num frames=3, end_frame=True, LR= 0.0000001, drop=0.3, dim_feedforward=2048, num_outputfeature=512, train=0.9, num_heads=4, num_layers=2/best_model'
    #model = get_model(best_model_dir)
    #create_audio_model_rdms("rsa_results", data_dir)
    #create_psts_rdms(model, data_dir, "rsa_results")
    #speaker_verification_model = EncoderClassifier.from_hparams(
      # source="speechbrain/spkrec-ecapa-voxceleb")
    #audio_rep = get_audio_model_embedding(speaker_verification_model, torchaudio.load("sample_13877_a_11.wav")[0])
    #print(audio_rep[-1][0].size())
    create_rsa_from_two_rdms("psts_model_audio_frames_rdm.xlsx", "psts",
                       "audio_model_audio_frames_rdm.xlsx", "audio_model_audio_frames2", "rsa_results")
    #plot_rsa("rsa_results/psts_audio_model_audio_frames2_full_corr_rsa.xlsx")
    #neptune = neptune.init_run(
     #   project="psts-through-unsupervised-learning/psts",
      #  api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzODRhM2YzNi03Nzk4LTRkZDctOTJiZS1mYjMzY2EzMDMzOTMifQ==")
    #print(get_psts_representation(model, dataset))