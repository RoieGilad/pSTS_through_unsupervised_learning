import math
from os import path

import torch
import torchvision.models as models
from torch import nn, Tensor


class PositionalEncoding(nn.Module):
    def __init__(self, positional_params: dict):
        super().__init__()
        self.model_type = 'PositionalEncoding'
        d_model = positional_params['d_model']
        self.dropout = nn.Dropout(p=positional_params['dropout'])

        position = torch.arange(positional_params['max_len']).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(positional_params['max_len'], 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """Arguments:
            x: Tensor, shape=[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_params, init_weights=True):
        super(TransformerDecoder, self).__init__()
        self.model_type = 'TransformerDecoder'
        self.positional_encoding = PositionalEncoding(
            decoder_params['positional_params'])
        self.d_model = decoder_params['hidden_dim']
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.d_model,
                                                   nhead=decoder_params['num_heads'],
                                                   dim_feedforward=decoder_params['dim_feedforward'],
                                                   batch_first =decoder_params['batch_first'])
        self.transformer_decoder = \
            nn.TransformerDecoder(decoder_layer,
                                  num_layers=decoder_params['num_layers'])
        self.linear = nn.Linear(self.d_model,
                                decoder_params['num_output_features'])
        mask = decoder_params['mask']
        self.register_buffer('mask', mask)

        if init_weights:
            self.init_weights()

    def forward(self, x, mask):
        x *= math.sqrt(self.d_model)
        x += self.positional_encoding(x)
        x = self.transformer_decoder(x, mask)
        x = self.linear(x)
        return x

    def init_weights(self):
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.linear.bias, 0.0)


class VideoDecoder(nn.Module):
    def __int__(self, model_params: dict, init_weights=True):
        super(VideoDecoder, self).__init__()
        self.model_type = 'VideoDecoder'
        self.num_frames = model_params['num_frames']
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features,
                                   model_params[
                                       'dim_resnet_to_transformer'])  # function  as the embedding layer
        self.decoder = TransformerDecoder(
            model_params['TransformerDecoder_params'])
        if init_weights:
            self.init_weights()

    def forward(self, frames):
        frames = torch.split(frames, self.num_frames, dim=0)
        frames = [self.resnet(frame) for frame in frames]
        frames = torch.stack(frames)
        frames = self.decoder(frames)
        return frames

    def init_weights(self):
        nn.init.normal_(self.resnet.fc.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.resnet.fc.bias, 0.0)

    def save_model(self, audio_dir):
        self.eval()
        resnet_path = path.join(audio_dir, "v_resnet.pt")
        transformer_path = path.join(audio_dir, "v_transformer.pt")
        hyperparams_path = path.join(audio_dir, "v_hyperparams.pt")
        torch.save(self.resnet.state_dict(), resnet_path)
        torch.save(self.decoder.state_dict(), transformer_path)
        torch.save(self.model_params, hyperparams_path)

    @classmethod
    def load_model(cls, audio_dir):
        resnet_path = path.join(audio_dir, "v_resnet.pt")
        transformer_path = path.join(audio_dir, "v_transformer.pt")
        hyperparams_path = path.join(audio_dir, "v_hyperparams.pt")
        model_params = torch.load(hyperparams_path)
        model = cls(model_params, False)
        model.resnet.load_state_dict(torch.load(resnet_path))
        model.decoder.load_state_dict(torch.load(transformer_path))
        return model


class AudioDecoder(nn.Module):
    def __int__(self, model_params: dict, init_weights=True):
        super(AudioDecoder, self).__init__()
        self.model_type = 'AudioDecoder'
        self.model_params = model_params
        self.num_frames = model_params['num_frames']
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features,
                                   model_params[
                                       'dim_resnet_to_transformer'])  # function  as the embedding layer
        self.decoder = TransformerDecoder(
            model_params['TransformerDecoder_params'])
        if init_weights:
            self.init_weights()

    def forward(self, frames):
        frames = torch.split(frames, self.num_frames, dim=0)
        frames = [self.resnet(frame) for frame in frames]
        frames = torch.stack(frames)
        frames = self.decoder(frames)
        return frames

    def init_weights(self):
        nn.init.normal_(self.resnet.fc.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.resnet.fc.bias, 0.0)

    def save_model(self, audio_dir):
        self.eval()
        resnet_path = path.join(audio_dir, "a_resnet.pt")
        transformer_path = path.join(audio_dir, "a_transformer.pt")
        hyperparams_path = path.join(audio_dir, "a_hyperparams.pt")
        torch.save(self.resnet.state_dict(), resnet_path)
        torch.save(self.decoder.state_dict(), transformer_path)
        torch.save(self.model_params, hyperparams_path)

    @classmethod
    def load_model(cls, audio_dir):
        resnet_path = path.join(audio_dir, "a_resnet.pt")
        transformer_path = path.join(audio_dir, "a_transformer.pt")
        hyperparams_path = path.join(audio_dir, "a_hyperparams.pt")
        model_params = torch.load(hyperparams_path)
        model = cls(model_params, False)
        model.resnet.load_state_dict(torch.load(resnet_path))
        model.decoder.load_state_dict(torch.load(transformer_path))
        return model


class PstsDecoder(nn.Module):
    def __int__(self, model_params: dict, init_weights=True):
        super(PstsDecoder, self).__init__()
        self.model_type = 'PstsEncoder'
        self.num_frames = model_params['num_frames']
        self.audio_decoder = AudioDecoder(
            model_params['audio_params']) if init_weights else None
        self.video_decoder = VideoDecoder(
            model_params['video_params']) if init_weights else None
        self.model_params = model_params
        if init_weights:
            self.audio_decoder.init_weights()
            self.video_decoder.init_weights()

    def forward_video(self, frames):
        return self.video_decoder(frames)

    def forward_audio(self, frames):
        return self.audio_decoder(frames)

    def forward(self, video, audio):
        return self.forward_video(video), self.forward_audio(audio)

    def save_model(self, path_to_save):
        self.eval()
        hyperparams_path = path.join(path_to_save, "hyperparams.pt")
        audio_path = path.join(path_to_save, "audio")
        video_path = path.join(path_to_save, "video")
        torch.save(self.model_params, hyperparams_path)
        self.audio_decoder.save_model(audio_path)
        self.video_decoder.save_model(video_path)

    @classmethod
    def load_model(cls, path_to_load):
        hyperparams_path = path.join(path_to_load, "hyperparams.pt")
        audio_dir = path.join(path_to_load, "audio")
        video_dir = path.join(path_to_load, "video")
        model_params = torch.load(hyperparams_path)
        model = cls(model_params, False)
        model.audio_decoder = AudioDecoder.load_model(audio_dir)
        model.video_decoder = VideoDecoder.load_model(video_dir)
        return model
