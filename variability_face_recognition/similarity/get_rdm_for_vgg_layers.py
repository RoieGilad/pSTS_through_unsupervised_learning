import os
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt

from rdm import load_data, get_embeddings, get_rdm, save_rdm_matrix_with_names


layers_dict = {'conv1': ['features.module.4', 802816],
               'conv2': ['features.module.9', 401408],
               'conv3': ['features.module.16', 200704],
               'conv4': ['features.module.23', 100352],
               'conv5': ['features.module.30', 25088],
               'fc6': ['classifier.4', 4096],
               'fc7': ['classifier.1', 4096],
               'fc8': ['classifier.6', 8749]}

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-data_dir", "--data_dir", dest="data_dir", help="folder with data")
    parser.add_argument("-model_path", "--model_path", dest="model_path", help="path to model weights", required=False)
    parser.add_argument("-output_path", "--output_path", dest="output_path", help="csv result dir", required=False)
    args = parser.parse_args()

    model_type = 'vgg_vggface2'

    data_paths_list, names_list = load_data(args.data_dir)
    for layer in layers_dict.keys():
        layer_name = layers_dict[layer][0]
        layer_size = layers_dict[layer][1]
        model_embeddings = get_embeddings(data_paths_list, model_type, args.model_path, perform_mtcnn=True,
                                          layer_name=layer_name, layer_size=layer_size)

        #save embeddings
        np.save(os.path.join(args.output_path, f'VGG16_{layer}_embeddings.npy'), model_embeddings)

        #save rdm
        # rdm_matrix = get_rdm(model_embeddings)
        # save_path = os.path.join(args.output_path, f'RDM_VGG16_{layer}.csv')
        # save_rdm_matrix_with_names(rdm_matrix, names_list, save_path)
        #
        # #save rdm image
        # plt.imshow(rdm_matrix)
        # plt.title(f'RDM {layer}')
        # plt.savefig(os.path.join(args.output_path, f'RDM_VGG16_{layer}.png'))
        # plt.close()
