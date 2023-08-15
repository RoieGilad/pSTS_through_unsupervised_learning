import math
import os
from os import path

import torch
import torchvision.models as models
from torch import nn, Tensor


class PositionalEncoding(nn.Module):
    def __init__(self, positional_params: dict):
        super(PositionalEncoding, self).__init__()
        self.model_type = 'PositionalEncoding'
        d_model = positional_params['d_model']
        self.dropout = nn.Dropout(p=positional_params['dropout'])

        position = torch.arange(0, positional_params['max_len']).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, positional_params['max_len'], d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        # pe = pe.squeeze(dim=1)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """Arguments:
            x: Tensor, shape=[seq_len, batch_size, embedding_dim]``
        """
        dim = 0 if len(x.shape) <= 2 else 1
        positional_encoding = self.pe[:, :x.size(dim), :]
        return x + self.dropout(positional_encoding)


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_params, init_weights=True):
        super(TransformerDecoder, self).__init__()
        self.model_type = 'TransformerDecoder'
        self.positional_encoding = PositionalEncoding(
            decoder_params['positional_params'])
        self.d_model = decoder_params['hidden_dim']
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.d_model,
                                                   nhead=decoder_params[
                                                       'num_heads'],
                                                   dim_feedforward=
                                                   decoder_params[
                                                       'dim_feedforward'],
                                                   batch_first=decoder_params[
                                                       'batch_first'])
        self.transformer_decoder = \
            nn.TransformerDecoder(decoder_layer,
                                  num_layers=decoder_params['num_layers'])
        self.linear = nn.Linear(self.d_model,
                                decoder_params['num_output_features'])
        mask = decoder_params['mask']
        self.register_buffer('mask', mask)

        if init_weights:
            self.init_weights()

    def forward(self, x):
        x *= math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        x = self.transformer_decoder(x, x, self.mask)
        x = self.linear(x)
        return x

    def init_weights(self):
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.linear.bias, 0.0)


class VideoDecoder(nn.Module):
    def __init__(self, model_params: dict, init_weights=True):
        super(VideoDecoder, self).__init__()
        self.use_end_frame = model_params['use_end_frame']
        self.use_decoder = model_params['use_decoder']
        self.model_type = 'VideoDecoder'
        self.num_frames = model_params['num_frames']
        self.dim_resnet_to_transformer = model_params[
            'dim_resnet_to_transformer']
        self.model_params = model_params
        self.resnet = models.resnet18(weights=None)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features,
                                   self.dim_resnet_to_transformer)  # function  as the embedding layer
        self.decoder = TransformerDecoder(
            model_params['TransformerDecoder_params'], self.use_end_frame) if self.use_decoder else None
        self.end_frame = nn.Parameter(torch.randn(3, 224, 224)) if self.use_decoder else None
        if init_weights:
            self.init_weights()

    def forward(self, frames):
        is_batched = len(frames.shape) > 4
        if is_batched:
            if self.use_end_frame:
                end_frame = self.end_frame.unsqueeze(0).repeat(frames.shape[0], 1, 1, 1, 1)
                frames = torch.cat((frames, end_frame), dim=1)
            bs, nf, c, h, w = frames.shape
            frames = frames.reshape(bs * nf, c, h, w)
        elif self.use_end_frame:
            frames = torch.cat((frames, self.end_frame), dim=0)
        frames = self.resnet(frames)
        if is_batched:
            frames = frames.reshape(bs, nf, self.dim_resnet_to_transformer)
        if self.use_decoder:
            frames = self.decoder(frames)
        return frames

    def init_weights(self):
        nn.init.normal_(self.resnet.fc.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.resnet.fc.bias, 0.0)

    def save_model(self, dir_to_save, device, distributed=False):
        self.eval()
        if not path.exists(dir_to_save):
            os.makedirs(dir_to_save, mode=0o777)
        resnet_path = path.join(dir_to_save, "v_resnet.pt")
        transformer_path = path.join(dir_to_save, "v_transformer.pt")
        hyperparams_path = path.join(dir_to_save, "v_hyperparams.pt")

        if not distributed and device.type == 'cuda':
            torch.save(self.resnet.state_dict(), resnet_path)
            if self.decoder:
                torch.save(self.decoder.state_dict(), transformer_path)
        else:
            torch.save(self.resnet.state_dict(), resnet_path)
            if self.decoder:
                torch.save(self.decoder.state_dict(), transformer_path)
        torch.save(self.model_params, hyperparams_path)

    def load_model(self, dir_to_load):
        hyperparams_path = path.join(dir_to_load, "v_hyperparams.pt")
        model_params = torch.load(hyperparams_path)
        self.__init__(model_params, False)
        self.load_model_weights(dir_to_load)

    def load_model_weights(self, dir_to_load):
        resnet_path = path.join(dir_to_load, "v_resnet.pt")
        transformer_path = path.join(dir_to_load, "v_transformer.pt")
        self.resnet.load_state_dict(torch.load(resnet_path))
        if self.use_decoder and self.decoder and os.path.exists(transformer_path):
            self.decoder.load_state_dict(torch.load(transformer_path))

    def load_resnet(self, dir_to_load):
        resnet_path = path.join(dir_to_load, "v_resnet.pt")
        self.resnet.load_state_dict(torch.load(resnet_path))


class AudioDecoder(nn.Module):
    def __init__(self, model_params: dict, init_weights=True):
        super(AudioDecoder, self).__init__()
        self.use_end_frame = model_params['use_end_frame']
        self.use_decoder = model_params['use_decoder']
        self.model_type = 'AudioDecoder'
        self.dim_resnet_to_transformer = model_params['dim_resnet_to_transformer']
        self.model_params = model_params
        self.num_frames = model_params['num_frames']
        self.resnet = models.resnet18(weights=None)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features,
                                   model_params[
                                       'dim_resnet_to_transformer'])  # function  as the embedding layer
        self.decoder = TransformerDecoder(
            model_params['TransformerDecoder_params'], self.use_end_frame) if self.use_decoder else None
        self.end_frame = nn.Parameter(torch.randn(3, 224, 224)) if self.use_end_frame else None
        if init_weights:
            self.init_weights()

    def forward(self, frames):
        is_batched = len(frames.shape) > 4
        if is_batched:
            if self.use_end_frame:
                end_frame = self.end_frame.unsqueeze(0).repeat(frames.shape[0], 1, 1, 1, 1)
                frames = torch.cat((frames, end_frame), dim=1)
            bs, nf, c, h, w = frames.shape
            frames = frames.reshape(bs * nf, c, h, w)
        elif self.use_end_frame:
            frames = torch.cat((frames, self.end_frame), dim=0)
        frames = self.resnet(frames)
        if is_batched:
            frames = frames.reshape(bs, nf, self.dim_resnet_to_transformer)
        if self.use_decoder:
            frames = self.decoder(frames)
        return frames

    def init_weights(self):
        nn.init.normal_(self.resnet.fc.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.resnet.fc.bias, 0.0)

    def save_model(self, dir_to_save, device, distributed=False):
        self.eval()
        if not path.exists(dir_to_save):
            os.makedirs(dir_to_save, mode=0o777)
        resnet_path = path.join(dir_to_save, "a_resnet.pt")
        transformer_path = path.join(dir_to_save, "a_transformer.pt")
        hyperparams_path = path.join(dir_to_save, "a_hyperparams.pt")

        if not distributed and device.type == 'cuda':
            torch.save(self.resnet.state_dict(), resnet_path)
            if self.decoder:
                torch.save(self.decoder.state_dict(), transformer_path)
        else:
            torch.save(self.resnet.state_dict(), resnet_path)
            if self.decoder:
                torch.save(self.decoder.state_dict(), transformer_path)
        torch.save(self.model_params, hyperparams_path)

    def load_model(self, dir_to_load):
        hyperparams_path = path.join(dir_to_load, "a_hyperparams.pt")
        model_params = torch.load(hyperparams_path)
        self.__init__(model_params, False)
        self.load_model_weights(dir_to_load)

    def load_model_weights(self, dir_to_load):
        resnet_path = path.join(dir_to_load, "a_resnet.pt")
        transformer_path = path.join(dir_to_load, "a_transformer.pt")
        self.resnet.load_state_dict(torch.load(resnet_path))
        if self.use_decoder and self.decoder and os.path.exists(transformer_path):
            self.decoder.load_state_dict(torch.load(transformer_path))

    def load_resnet(self, dir_to_load):
        resnet_path = path.join(dir_to_load, "a_resnet.pt")
        self.resnet.load_state_dict(torch.load(resnet_path))


class PstsDecoder(nn.Module):
    def __init__(self, model_params: dict = None, init_weights=True,
                 use_end_frame=True, use_decoder=True):
        super(PstsDecoder, self).__init__()
        self.model_type = 'PstsEncoder'
        self.num_frames = model_params['num_frames'] if model_params else None
        self.audio_decoder = AudioDecoder(model_params['audio_params'],
                                          init_weights) if model_params else None
        self.video_decoder = VideoDecoder(model_params['video_params'],
                                          init_weights) if model_params else None
        self.model_params = model_params

    def forward_video(self, frames):
        return self.video_decoder(frames)

    def forward_audio(self, frames):
        return self.audio_decoder(frames)

    def forward(self, video, audio):
        return self.forward_video(video), self.forward_audio(audio)

    def save_model(self, path_to_save, device, distributed=False):
        self.eval()
        if not path.exists(path_to_save):
            os.makedirs(path_to_save, mode=0o777)
        hyperparams_path = path.join(path_to_save, "psts_hyperparams.pt")
        torch.save(self.model_params, hyperparams_path)
        self.audio_decoder.save_model(path_to_save, device, distributed)
        self.video_decoder.save_model(path_to_save, device, distributed)

    def load_model(self, path_to_load):
        hyperparams_path = path.join(path_to_load, "psts_hyperparams.pt")
        model_params = torch.load(hyperparams_path)
        self.__init__(model_params, False)
        self.audio_decoder.load_model_weights(path_to_load)
        self.video_decoder.load_model_weights(path_to_load)

    def set_use_decoder(self, state):
        self.video_decoder.use_decoder = state
        self.audio_decoder.use_decoder = state

    def set_use_end_frame(self, state):
        self.video_decoder.use_end_frame = state
        self.audio_decoder.use_end_frame = state

    def load_resnet(self, path_to_load):
        self.audio_decoder.load_resnet(path_to_load)
        self.video_decoder.load_resnet(path_to_load)
