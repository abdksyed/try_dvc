import torch
import torch.nn as nn
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Save the Feature Map as a numpy array(.npy)
@torch.no_grad()
def save_feature_map(input_data, feature_extractor, out_path, mode='train', logger=None, device=device, force=False):
    '''
    It takes input images and save the feature map of the images in the numpy array.
    The feature maps are attained by passing the input images through the feature extractor.

    :param input_data: A tensor of shape (N, C, H, W)
    :param feature_extractor: A pytorch model
    :param out_path: The path to save the feature map
    :param mode: 'train' or 'test'
    :param device: 'cpu' or 'cuda'
    '''
    if (out_path/f'{mode}_bottleneck_features.npy').exists() and not force:
        if logger is not None:
            logger.info(f'{mode} bottleneck features already exist. Skipping...')
        return
    final_tensor = []
    label_tensor = []
    feature_extractor = feature_extractor.to(device)
    for idx, (data, label) in enumerate(input_data):
        data = data.to(device)
        features = feature_extractor(data)
        final_tensor.append(features)
        label_tensor.append(label)
    final_tensor = torch.cat(final_tensor)
    label_tensor = torch.cat(label_tensor)

    final_tensor = final_tensor.cpu().detach().numpy()
    label_tensor = label_tensor.cpu().detach().numpy().astype(np.float32)

    if logger is not None:
        logger.info(f'Saving {mode} bottleneck features of shape {final_tensor.shape}...')
        
    np.save(open(f'{out_path}/{mode}_bottleneck_features.npy', 'wb'), final_tensor)
    np.save(open(f'{out_path}/{mode}_labels.npy', 'wb'), label_tensor)



