# Get Similarity Matrix
The scripts here can be used to get a similarity matrix for all images in a given dataset. 
The matrix will represent the cosine similarity between the embeddings of each pair of images.

These embeddings are obtained by passing the images through a pre-trained model, and extracting the output of the last layer before the classification layer 
(In VGG16 for example, this would be fc7).

This module allows to experiment with different models, detailed in the next section.

### Models

#### *1. VGG architecure, trained on VGGFace2 dataset*

Pre-trained on 8749 identities from the VGGFace2 dataset (Trained by Idan :crown:)

When using this model, weights should be provided.
Ask Idan or Shiri for the weights file (shirialmog1@gmail.com)

#### *2. VGG architecure, trained on imagenet dataset*
   
Model available online, no weights needed.

#### *3. InceptionResNet architecture, pretrained on vggface2 or casia-webface*
   
See documentation here: https://github.com/timesler/facenet-pytorch


### Data 

The data should be organized in the following way:

- main_folder
  - id1
      - img1
      - img2
      - ...
  - id2
      - img1
      - img2
      - ...


### Usage

To get the similarity matrix with the default model (resnet pretrained on vggface2), run the following command:
```bash
python similarity/rdm.py --data_dir <path_to_data> --output_path <path_to_results_dir>
```
To experiment with different models, add the arguments `--model_type` with one of the following options:
- vgg_vggface2 
- vgg_imagenet 
- resnet_vggface2 (default)
- resnet_casia

See Models section above for more details.

Note: if type chosen is vgg_vggface2, `--model_path` should be provided with local path to weights.

```bash
python similarity/rdm.py --data_dir <path_to_data> --output_path <path_to_results_dir> --model_type vgg_vggface2 --model_path <path_to_weights>
```
