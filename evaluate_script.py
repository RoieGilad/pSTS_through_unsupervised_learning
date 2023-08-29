import time

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
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
import seaborn as sns


def run_and_compare(model, v1, a1, v2, a2, device, i, j):
    videos = torch.stack([v1, v2])
    audios = torch.stack([a1, a2])
    videos = videos.to(device)
    audios = audios.to(device)
    encode_videos, encode_audios = model(videos, audios)
    encode_v1, encode_v2 = encode_videos[0], encode_videos[1]
    encode_a1, encode_a2 = encode_audios[0], encode_audios[1]
    if i == j:
        C11 = pairwise_cosine_similarity(encode_v1, encode_a1)
        return C11, None, None
    else:
        C12 = pairwise_cosine_similarity(encode_v1, encode_a2)
        C21 = pairwise_cosine_similarity(encode_v2, encode_a1)

    return None, C12, C21


def collect_comparison(model, doc, dataset, max_samples=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_samples = len(dataset) if not max_samples else max_samples
    model = model.to(device)
    model.eval()
    for i in tqdm(range(max_samples)):
        for j in range(i, max_samples):
            v1, a1, l1 = dataset[i]
            v2, a2, l2 = dataset[j]
            C11, C12, C21 = run_and_compare(model, v1, a1, v2, a2, device, i, j)

            if i == j:
                doc["cosine/same_frame_similarity"].extend(
                    C11.diag().flatten().tolist())
                rows, cols = C11.shape
                sequence_similarity = [C11[a][b] for b in range(cols) for a in
                                       range(rows) if
                                       a != b]
                doc["cosine/sequence_similarity"].extend(sequence_similarity)
            else:
                if l1 == l2:
                    category = "cosine/in_identity_similarity"
                else:
                    category = "cosine/between_identities_similarity"
                doc[category].extend(C12.flatten().tolist())
                doc[category].extend(C21.flatten().tolist())
    print("cosine/same_frame_similarity", doc["cosine/same_frame_similarity"])
    print("cosine/sequence_similarity", doc["cosine/sequence_similarity"])
    print("cosine/in_identity_similarity", doc["cosine/in_identity_similarity"])
    print("cosine/between_identities_similarity",
          doc["cosine/between_identities_similarity"])


def fast_run_and_compare(model, videos, audios, device):
    videos = torch.stack([videos])
    audios = torch.stack([audios])
    videos = videos.to(device)
    audios = audios.to(device)
    encode_videos, encode_audios = model(videos, audios)
    return encode_videos, encode_audios


def fast_collect_comparison(model, doc, dataset, max_samples=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_samples = len(dataset) if not max_samples else max_samples
    model = model.to(device)
    model.eval()
    outputs = [[None, None, None] for _ in range(max_samples)]

    for i in tqdm(range(max_samples)):
        v1, a1, l1 = dataset[i]
        encode_v1, encode_a1 = fast_run_and_compare(model, v1, a1, device)
        outputs[i] = [encode_v1[0], encode_a1[0], l1]

    for i in tqdm(range(max_samples)):
        same_frame_similarity, sequence_similarity,in_identity_similarity, \
            between_identities_similarity = [], [], [], []
        for j in range(i, max_samples):
            encode_v1, encode_a1, l1 = outputs[i]
            encode_v2, encode_a2, l2 = outputs[j]
            if i == j:
                C11 = pairwise_cosine_similarity(encode_v1, encode_a1)
                same_frame_similarity.extend(C11.diag().flatten().tolist())
                rows, cols = C11.shape
                sequence_similarity.extend([C11[a][b] for b in range(cols) for a
                                       in range(rows) if
                                       a != b])
            else:
                C12 = pairwise_cosine_similarity(encode_v1, encode_a2)
                C21 = pairwise_cosine_similarity(encode_v2, encode_a1)
                if l1 == l2:
                    in_identity_similarity.extend(C12.flatten().tolist())
                    in_identity_similarity.extend(C21.flatten().tolist())
                else:
                    between_identities_similarity.extend(C12.flatten().tolist())
                    between_identities_similarity.extend(C21.flatten().tolist())

        doc["cosine/same_frame_similarity"].extend(same_frame_similarity)
        doc["cosine/sequence_similarity"].extend(sequence_similarity)
        doc["cosine/in_identity_similarity"].extend(in_identity_similarity)
        doc["cosine/between_identities_similarity"].extend(between_identities_similarity)




def get_model(path_to_load, num_frames=3):
    batch_size = 54
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
                    }
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


def get_dataset(path_to_dataset_dir, num_frames=3):
    transforms_dict = {'a_frame_transform': du.val_a_frame_transformer,
                       'a_batch_transform': du.val_audio_transformer,
                       'v_frame_transform': du.val_v_frame_transformer,
                       'v_batch_transform': du.val_video_transformer}
    combined_dataset = CombinedDataset(path_to_dataset_dir,
                                       du.get_label_path(path_to_dataset_dir),
                                       transforms_dict,
                                       num_frames=num_frames,
                                       test=True,
                                       step_size=1)
    return combined_dataset


def run_validation(model, doc, max_samples, dataset, batch_size=54):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_samples = len(dataset) if not max_samples else max_samples
    model = model.to(device)
    model.eval()
    running_vloss = 0.0
    validation_dataloader = torch.utils.data.DataLoader(dataset, batch_size,
                                                        shuffle=True)
    Loss = pstsLoss()
    with torch.no_grad():  # Disable gradient computation
        for i, vdata in tqdm(enumerate(validation_dataloader)):
            batch_loss = run_one_batch_psts(Loss, model, vdata, False, None,
                                            device)
            running_vloss += batch_loss
            doc['test/every_batch'].append(batch_loss)
            if i * batch_size >= max_samples:
                break
    avg_vloss = running_vloss / (i + 1)
    doc['test/avg_test_loss'].append(avg_vloss)

def run_gpu_test():
    seed = 42
    torch.manual_seed(seed)
    test_dir = r'dataset/test/160k_test_500ms'
    best_model_dir = r'models/check transformer whole DS, no gradient BS= 54, num frames=3, end_frame=True, LR= 0.0000001, drop=0.3, dim_feedforward=2048, num_outputfeature=512, train=0.9, num_heads=4, num_layers=2/best_model'

    dataset = get_dataset(test_dir)
    model = get_model(best_model_dir)
    neptune = neptune.init_run(
        project="psts-through-unsupervised-learning/psts",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzODRhM2YzNi03Nzk4LTRkZDctOTJiZS1mYjMzY2EzMDMzOTMifQ==")

    start = time.time()
    fast_collect_comparison(model, neptune, dataset, 100)
    end = time.time()
    print("compare took: ", end-start)

    start = time.time()
    run_validation(model, neptune, 100)
    end = time.time()
    print("run_validation took: ", end-start)


def make_distribution_plot(points, title):
    if np.isnan(points).any():
        points = points[~np.isnan(points)]
    mean_value = np.mean(points)
    std_value = np.std(points)
    print(title, f'mean: {mean_value}', f'std: {std_value}')
    sns.kdeplot(points, shade=True, common_norm=True)
    if title != "loss on 54 batch size on 5000 test samples":
        plt.xlim(-1, 1)
    # plt.hist(points, bins=min(100000, len(points)), density=True, alpha=0.75,
    #          histtype='barstacked', color='b')
    plt.title(f'Distribution of {title}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.text(0.1, 0.9, f'Mean: {mean_value:.2f}', transform=plt.gca().transAxes)
    plt.text(0.1, 0.85, f'Std Dev: {std_value:.2f}',
             transform=plt.gca().transAxes)
    plt.show()

def create_distribution_plots(dir_to_csv, concatenate=False):
    data = pd.read_excel(dir_to_csv)
    if not concatenate:
        for column_name in data.columns:
            make_distribution_plot(data[column_name].to_numpy(), column_name)
    else:
        points, title = None, ""
        first = True
        for column_name in data.columns:
            if first:
                first = False
                title = column_name
                points = data[column_name].to_numpy()
            else:
                points = np.concatenate([points, data[column_name].to_numpy()])
        make_distribution_plot(points, title)



if __name__ == '__main__':
    create_distribution_plots(r'cosine similarity within identity.xlsx', True)
    create_distribution_plots(r'three metrics.xls', False)
