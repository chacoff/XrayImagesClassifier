import seaborn as sns
from torchvision import transforms, datasets, models
import torch
from torch import optim, cuda
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
from torchsummary import summary
import numpy as np
import pandas as pd
import random
import os
from PIL import Image
import cv2
from timeit import default_timer as timer
from tqdm import tqdm
from utils import *
import matplotlib.pyplot as plt
import warnings
plt.rcParams['font.size'] = 10
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

args = {
    'data_test': 'Covid19\\test',
    'weights': 'weights\\resnet50\\inference.pt',
    'random': True,
    'save_png': True,
}

if args['random'] is True:
    folders = os.listdir(args['data_test'])
    folder = os.path.join(args['data_test'], random.choice([x for x in folders]))
    image = random.choice([x for x in os.listdir(folder) if os.path.isfile(os.path.join(folder, x))])
    Xray_image = os.path.join(folder, image)
else:  # choose manually an image from dataset/test
    image = '\\Covid\\auntminnie-d-2020_01_28_23_51_6665_2020_01_28_Vietnam_coronavirus.jpeg'
    Xray_image = args['data_test'] + image

real_label = Xray_image.split('\\')[-2]
print(f'[INFO] image Label: {real_label}')
weight = args['weights']
weight_name = weight.split('\\')[-1]
print(f'[INFO] loading weight: {weight_name}')

if cuda.is_available():
    load_Torch = 'cuda:'+str(torch.cuda.current_device())
    print(f'[INFO] GPU: {torch.cuda.get_device_name(load_Torch)}')
else:
    load_Torch = 'cpu'

model = torch.load(weight, map_location=load_Torch)  # map_location could be remove in presence of GPU
model.eval()

ok_prediction = display_prediction2(Xray_image, model, real_label, topk=3, save=args['save_png'])

if ok_prediction:
    print(f'[INFO] The neural network has classified correctly the image: {Xray_image}')
else:
    print(f'[INFO] The neural network has misclassified the image: {Xray_image}')