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


def run_and_compare(model, v1, a1, v2, a2, device, i, j):
    videos = torch.stack([v1, v2])
    audios = torch.stack([a1, a2])
    videos = videos.to(device)
    audios = audios.to(device)
    encode_videos, encode_audios = model(videos, audios)
    encode_v1, encode_v2 = encode_videos[0], encode_videos[2]
    encode_a1, encode_a2 = encode_audios[0], encode_audios[2]
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
        for j in tqdm(range(i + 1, max_samples)):
            v1, a1, l1 = dataset[i]
            v2, a2, l2 = dataset[j]
            C11, C12, C21 = run_and_compare(model, v1, a1, v2, a2, device, i, j)

            if i == j:
                doc["data/same_frame_similarity"].extend(C11.diag().tolist())
                rows, cols = C11.shape
                sequence_similarity = [C11[a][b] for b in cols for a in rows if
                                       a != b]
                doc["data/sequence_similarity"].extend(sequence_similarity)
            else:
                if l1 == l2:
                    category = "data/in_identity_similarity"
                else:
                    category = "data/between_identities_similarity"
                doc[category].extend(C12.tolist())
                doc[category].extedn(C21.tolist())
    print("data/same_frame_similarity", doc["data/same_frame_similarity"])
    print("data/sequence_similarity", doc["data/sequence_similarity"])
    print("data/in_identity_similarity", doc["data/in_identity_similarity"])
    print("data/between_identities_similarity",
          doc["data/between_identities_similarity"])


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


def run_validation(model, doc, max_samples, batch_size=54):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_samples = len(dataset) if not max_samples else max_samples
    model = model.to(device)
    model.eval()
    running_vloss = 0.0
    validation_dataloader = torch.utils.data.DataLoader(dataset, batch_size,
                                                        shuffle=True)
    Loss = pstsLoss()
    with torch.no_grad():  # Disable gradient computation
        for i, vdata in enumerate(validation_dataloader):
            batch_loss = run_one_batch_psts(Loss, model, vdata, False, None,
                                            device)
            running_vloss += batch_loss
            doc['test/every_batch'].append(batch_loss)
            if i == max_samples:
                break
    avg_vloss = running_vloss / (i + 1)
    doc['test/abg_test_loss'].append(avg_vloss)


if __name__ == '__main__':
    seed = 42
    torch.manual_seed(seed)
    test_dir = r'dataset/test/160k_test_500ms'
    best_model_dir = r'models/check transformer whole DS, no gradient BS= 54, num frames=3, end_frame=True, LR= 0.0000001, drop=0.3, dim_feedforward=2048, num_outputfeature=512, train=0.9, num_heads=4, num_layers=2/best_model'

    dataset = get_dataset(test_dir)
    model = get_model(best_model_dir)
    neptune = neptune.init_run(
        project="psts-through-unsupervised-learning/psts",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzODRhM2YzNi03Nzk4LTRkZDctOTJiZS1mYjMzY2EzMDMzOTMifQ==")

    collect_comparison(model, neptune, dataset, 10)

    run_validation(model, neptune, 10)
