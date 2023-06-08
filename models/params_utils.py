
def init_positional_Encoding_params(d_model, dropout, max_len=100):
    return {'d_model': d_model, 'dropout': dropout, 'max_len': max_len}


def init_transformer_decoder_params(hidden_dim, num_heads, dim_feedforward,
                                    batch_first, num_layers,
                                    num_output_features, mask, dropout,
                                    max_len=100):
    params = {'hidden_dim': hidden_dim,
              'num_heads': num_heads,
              'dim_feedforward': dim_feedforward,
              'batch_first': batch_first,
              'num_layers': num_layers,
              'num_output_features': num_output_features,
              'mask': mask,
              'positional_params': init_positional_Encoding_params(hidden_dim,
                                                                   dropout,
                                                                   max_len)
              }
    return params


def init_Video_decoder_params(num_frames, dim_resnet_to_transformer,
                              num_heads, dim_feedforward, batch_first,
                              num_layers, num_output_features, mask,
                              dropout, max_len=100):
    params = {'num_frames': num_frames,
              'dim_resnet_to_transformer': dim_resnet_to_transformer,
              'TransformerDecoder_params': init_transformer_decoder_params(
                  dim_resnet_to_transformer, num_heads, dim_feedforward,
                  batch_first, num_layers, num_output_features, mask, dropout,
                  max_len)}
    return params


def init_audio_decoder_params(num_frames, dim_resnet_to_transformer,
                              num_heads, dim_feedforward, batch_first,
                              num_layers, num_output_features, mask,
                              dropout, max_len=100):
    params = {'num_frames': num_frames,
              'dim_resnet_to_transformer': dim_resnet_to_transformer,
              'TransformerDecoder_params': init_transformer_decoder_params(
                  dim_resnet_to_transformer, num_heads, dim_feedforward,
                  batch_first, num_layers, num_output_features, mask, dropout,
                  max_len)}
    return params


def init_psts_decoder_params(num_frames, video_params, audio_params):
    video_params['num_frames'] = audio_params['num_frames'] = num_frames
    params = {'num_frames': num_frames, 'video_params': video_params,
              'audio_params': audio_params}
    return params


example_audio = init_audio_decoder_params(num_frames=, dim_resnet_to_transformer=,
                              num_heads=, dim_feedforward=, batch_first=,
                              num_layers=, num_output_features=, mask=,
                              dropout=, max_len=100)

example_video = init_Video_decoder_params(num_frames=, dim_resnet_to_transformer=,
                              num_heads=, dim_feedforward=, batch_first=,
                              num_layers=, num_output_features=, mask=,
                              dropout=, max_len=100)

example_psts = init_psts_decoder_params(num_frames=, example_video, example_audio)