from torchmetrics.functional import pairwise_cosine_similarity
import torch
from torch import nn, Tensor


class pstsLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, encode_videos, encode_audios):
        """ 1) flat encode_videos/encode_audios to a matrix s.t.
        row[i] = the (i % num_frames)th frame in the (i//num_frames)th video/audio
        2) calculate the cosine_similarity matrix by calculate the
        cosine similarity between each pair of rows in the flatten tensors
        3) exp the matrix
        4) take the diag of the matrix, the vector sum rows\columns of the matrix
        5) calculate the log of diag/vector sum rows (columns)
        6) return the minus of sum of both vectors"""
        batch_size, num_frames, encoded_vector_size = encode_audios.shape
        flat_encode_videos = encode_videos.view(batch_size * num_frames,
                                                encoded_vector_size)
        flat_encode_audios = encode_audios.view(batch_size * num_frames,
                                                encoded_vector_size)
        C = pairwise_cosine_similarity(flat_encode_videos, flat_encode_audios)
        C = torch.exp(C)
        main_diag = torch.diag(C)
        rows_softmax_denominator = torch.sum(C, dim=1)
        columns_softmax_denominator = torch.sum(C, dim=0)
        rows_softmax_vals = torch.log(main_diag / rows_softmax_denominator)
        columns_softmax_vals = torch.log(main_diag / columns_softmax_denominator)
        return -1 * (torch.sum(rows_softmax_vals + columns_softmax_vals))
