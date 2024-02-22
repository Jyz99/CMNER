# _*_ coding :utf-8 _*_
# time: 2022/10/19/019 19:44:37
import os
import time
import argparse

import torch
import torch.nn as nn
from torchvision import transforms
import resnet.resnet as resnet
from resnet.resnet_utils import myResnet

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from PIL import Image
import numpy as np

def image_process(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--fine_tune_cnn', action='store_true', help='fine tune pre-trained CNN if True')
    parser.add_argument('--crop_size', type=int, default=224, help='crop size of image')
    parser.add_argument('--resnet_root', default='./resnet', help='path the pre-trained cnn models')
    parser.add_argument("--feature_file", default="img_vgg_features.pt", type=str, help="Filename for preprocessed image features")

    args = parser.parse_args()

    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    net = getattr(resnet, 'resnet152')()
    net.load_state_dict(torch.load(os.path.join(args.resnet_root, 'resnet152.pth')))
    encoder = myResnet(net, args.fine_tune_cnn, device)

    encoder.to(device)
    encoder.eval()

    input_img_directory = '/home/jiyuanze/UMT/data/weibo_images'

    img_features = {}

    transform = transforms.Compose([
        transforms.RandomCrop(args.crop_size),  # args.crop_size, by default it is set to be 224
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    for img_name in os.listdir(input_img_directory):
        image_path = os.path.join(input_img_directory, img_name)
        try:
            img_feat = image_process(image_path, transform)
            img_feat = img_feat.unsqueeze(0)
            img_feat = img_feat.to(device)
            with torch.no_grad():
                imgs_f, img_mean, img_att = encoder(img_feat)

            img_att = img_att.squeeze(0)
            img_features[img_name.split('.')[0]] = img_att.to('cpu')
        except:
            print(image_path + ' has problem!')


    torch.save(img_features, '/home/jiyuanze/UMT/data/weibo/resnet_img_features_motley.pth')
