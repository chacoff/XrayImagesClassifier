import seaborn as sns
from torchvision import transforms, datasets, models
import torch
from torch import optim, cuda
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
from torchsummary import summary
import numpy as np
import pandas as pd
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

# Data location
datadir = os.path.join('d:\\', 'AlienwareCoding', '4_CovidNet', 'Covid19')
traindir = os.path.join(datadir, 'train')
validdir = os.path.join(datadir, 'test')  # valid
testdir = os.path.join(datadir, 'test')
save_file_name = 'weights\\best.pt'
checkpoint_path = 'weights\\chkpt.pth'
# Training Parameters
adam_opt = False
store_checkpoint = False
show_summary = True
pre_model = 'resnet50'  # vgg16 or resnet50
lr = 0.001  # 0.0001 for Adam
lr_plateau = 0.1  # decay learning rate factor
weight_decay = 0.005
momentum = 0.900  # for SGD
batch_size = 8
epochs_train = 51
epochs_stop = 10

if cuda.is_available():  # Number of gpus and if to train on a gpu
    gpu_count = cuda.device_count()
    gpu_id = cuda.current_device()
    gpu_name = cuda.get_device_name(gpu_id)
    if gpu_count > 1:
        multi_gpu = True
    else:
        multi_gpu = False
    print(f'[INFO] Pre-trained model: {pre_model}')
    print(f'[INFO] GPU: {gpu_count} {gpu_name} detected')
    print(f'[INFO] Multi GPU: {multi_gpu}')
    if adam_opt is True:
        print(f'[INFO] Optimizer: ADAM, learning rate: {lr}, batch size: {batch_size}, epochs: {epochs_train}, '
              f'lr on plateau{lr_plateau}, epochs early stop: {epochs_stop}')
    else:
        print(f'[INFO] Optimizer: SGD, lr: {lr}, momentum: {momentum}, decay: {weight_decay}, epochs: {epochs_train}, '
              f'lr on plateau: {lr_plateau}, batch size: {batch_size},  epochs early stop: {epochs_stop}')
else:
    print(f'[INFO] Train on GPU: {cuda.is_available()}')
    multi_gpu = False

cv2.waitKey(20)
# DATA EXPLORATION: Empty lists
categories, img_categories, n_train, n_valid, n_test, hs, ws = ([], [], [], [], [], [], [])

total_train = sum(len(files) for _, _, files in os.walk(traindir))  # Iterate through each category to count images
pbar = tqdm(total=total_train, desc='Loading training images')  # 383 images, train + test * 2 / 251 images for training
for d in os.listdir(traindir):
    categories.append(d)
    # Number of each image
    train_imgs = os.listdir(os.path.join(traindir, d))
    valid_imgs = os.listdir(os.path.join(validdir, d))
    test_imgs = os.listdir(os.path.join(testdir, d))
    n_train.append(len(train_imgs))
    n_valid.append(len(valid_imgs))
    n_test.append(len(test_imgs))

    # Find stats for train images
    for i in train_imgs:
        img_categories.append(d)
        img = cv2.imread(os.path.join(traindir, d, i))  # img = Image.open(os.path.join(traindir, d, i))
        img_array = np.array(img)
        pbar.update(1)
        # Shape
        hs.append(img_array.shape[0])
        ws.append(img_array.shape[1])
pbar.close()
# print('Diseases: %s' % categories)

# Dataframe of categories
cat_df = pd.DataFrame({'category': categories,
                       'n_train': n_train,
                       'n_valid': n_valid,
                       'n_test': n_test}).\
                        sort_values('category')

cat_df.sort_values('n_train', ascending=False, inplace=True)
cat_df.head()
cat_df.tail()

# Dataframe of training images
image_df = pd.DataFrame({
    'category': img_categories,
    'height': hs,
    'width': ws
})

# DATA AUGMENTATION: Image transformations standardized for ImageNet
image_transforms = {
    # Train uses data augmentation
    'train':
    transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.2), ratio=(0.75, 1.333), interpolation=2),
        transforms.RandomRotation(degrees=16),
        transforms.ColorJitter(brightness=0.03, contrast=0.05, saturation=0.02, hue=0.3),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.CenterCrop(size=224),  # Imagenet standards
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])  # Imagenet standards
    ]),
    # Validation does not use augmentation
    'val':
    transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    # Test does not use augmentation
    'test':
    transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


# Datasets from each folder: DATA ITERATORS
data = {
    'train':
    datasets.ImageFolder(root=traindir, transform=image_transforms['train']),  # TORCHVISION.DATASETS.ImageFolder
    'val':
    datasets.ImageFolder(root=validdir, transform=image_transforms['val']),
    'test':
    datasets.ImageFolder(root=testdir, transform=image_transforms['test'])
}

# Dataloader iterators
dataloaders = {
    'train': DataLoader(data['train'], batch_size=batch_size, shuffle=True),
    'val': DataLoader(data['val'], batch_size=batch_size, shuffle=True),
    'test': DataLoader(data['test'], batch_size=batch_size, shuffle=True)
}

trainiter = iter(dataloaders['train'])
features, labels = next(trainiter)  # features.shape = (batch_size, color_channels, height, width)

n_classes =len(data['train'].classes) # len(cat_df)
print(f'[INFO] {n_classes} different classes.')


# PRE TRAINED MODEL
model = get_pretrained_model(pre_model, n_classes, multi_gpu)

# final output will be log probabilities for the Negative Log Likelihood Loss
total_params = sum(p.numel() for p in model.parameters())
print(f'[INFO] Total Parameters: {total_params:,}')
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'[INFO] Training Parameters: {total_trainable_params:,}\n')

if show_summary:
    if multi_gpu:
        summary(model.module, input_size=(3, 224, 224), batch_size=batch_size, device='cuda')
        if pre_model == 'resnet50':
            print(model.fc)
        else:
            print(model.module.classifier[6])
    else:
        summary(model, input_size=(3, 224, 224), batch_size=batch_size, device='cuda')  # input_size=channels,height,width
        if pre_model == 'resnet50':
            print(model.fc)
        else:
            print(model.classifier[6])

# mapping of classes to indexes
model.class_to_idx = data['train'].class_to_idx
model.idx_to_class = {idx: class_ for class_, idx in model.class_to_idx.items()}

# print(list(model.idx_to_class.items()))  # print the hot-encoding

# TRAINING - LOSS - OPTIMIZER
criterion = nn.CrossEntropyLoss()  # nn.NLLLoss() negative log likelihood loss
if adam_opt is True:
    optimizer = optim.Adam(model.parameters(), lr=lr)
else:
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=lr_plateau)  # Decay by gamma every 7 epochs

if store_checkpoint is True:
    model, optimizer = load_checkpoint(path=checkpoint_path)

model, history = train(
    model,
    criterion,
    optimizer,
    dataloaders['train'],
    dataloaders['val'],
    exp_lr_scheduler,
    save_file_name=save_file_name,
    max_epochs_stop=epochs_stop,
    n_epochs=epochs_train,
    print_every=1)

# save_checkpoint(model, path=checkpoint_path)
torch.save(model, 'weights\\inference.pt')  # quick save for fast inference
all_plots(cat_df, history, where='data')
