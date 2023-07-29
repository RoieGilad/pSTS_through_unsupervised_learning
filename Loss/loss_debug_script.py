import torch
from Loss.pstsLoss import pstsLoss


if __name__ == '__main__':
    batch_size = 2
    num_frames = 2
    encoded_vector_size = 4

    # Create example tensors
    encode_videos = torch.tensor([[[1, 0, 0, 0], [0, 1, 0, 0]],
                                  [[0, 0, 1, 0], [0, 0, 0, 1]]], dtype=torch.float32)
    encode_audios = torch.tensor(
        [[[25, 26, 27, 28], [29, 30, 31, 32], [33, 34, 35, 36]],
         [[37, 38, 39, 40], [41, 42, 43, 44], [45, 46, 47, 48]]],
        dtype=torch.float32)

    offir = torch.tensor(
        [[[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
         [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]],
        dtype=torch.float32)
    print(encode_videos)
    print()
    print(encode_audios)
    print()
    loss = pstsLoss()
    # pairwise_similarity_gpt = loss.create_all_c_j_gpt(encode_videos, encode_audios)
    pairwise_similarity = loss(encode_videos, encode_videos)
    print("Ours:")
    print(pairwise_similarity)
    print("GPT:")
    # print(pairwise_similarity_gpt)