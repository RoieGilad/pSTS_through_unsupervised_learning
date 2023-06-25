from torchmetrics.functional import pairwise_cosine_similarity
import torch
from torch import nn
import torch.nn.functional as F


class pstsLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, encode_videos, encode_audios):
        """ 1) flat encode_videos/encode_audios to a matrix s.t.
        row[i] = the (i % num_frames)th frame in the (i//num_frames)th video/audio
        2) calculate the cosine_similarity matrix by calculate the
        cosine similarity between each pair of rows in the flat tensors
        3) calculate the softmax matrix by column\rows and take the diagonal of
        each output.
        5) calculate the log of diag/vector each softmax vector
        6) return the minus of the mean of both vectors"""
        batch_size, num_frames, encoded_vector_size = encode_audios.shape
        flat_encode_videos = encode_videos.view(batch_size * num_frames,
                                                encoded_vector_size)
        flat_encode_audios = encode_audios.view(batch_size * num_frames,
                                                encoded_vector_size)
        C = pairwise_cosine_similarity(flat_encode_videos, flat_encode_audios)
        row_softmax_vector = torch.log(torch.diag(F.softmax(C, dim=1)))
        columns_softmax_vector = torch.log(torch.diag(F.softmax(C, dim=0)))
        return -1 * (torch.mean(row_softmax_vector + columns_softmax_vector))
