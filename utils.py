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
import os, sys
from PIL import Image
import cv2
from timeit import default_timer as timer
from tqdm import tqdm
from utils import *
import matplotlib.pyplot as plt
from cv2_plt_imshow import cv2_plt_imshow, plt_format
import warnings
plt.rcParams['font.size'] = 10
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

'''  # Example of Data Augmentation
# ex_img = cv2.imread(os.path.join(traindir, 'Pneumonia', '032.jpeg'))
ex_img = Image.open(os.path.join(traindir, 'Pneumonia', '032.jpeg'))  # with PIL
t = image_transforms['train']
plt.figure(figsize=(24, 24))

for i in range(16):
    ax = plt.subplot(4, 4, i + 1)
    _ = imshow_tensor(t(ex_img), ax=ax)
plt.tight_layout()
plt.savefig('data\\augmentation.png')  # plt.show()
'''


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def display_category(model, category, n=4):
    """Display predictions for a category    
    """
    category_results = results.loc[results['class'] == category]
    print(category_results.iloc[:, :6], '/n')

    images = np.random.choice(
        os.listdir(testdir + category + '/'), size=4, replace=False)

    for img in images:
        display_prediction(testdir + category + '/' + img, model, 5)


def evaluate(model, test_loader, criterion, topk=(1, 5)):
    """Measure the performance of a trained PyTorch model
    Params
    --------
        model (PyTorch model): trained cnn for inference
        test_loader (PyTorch DataLoader): test dataloader
        topk (tuple of ints): accuracy to measure
    Returns
    --------
        results (DataFrame): results for each category
    """
    classes = []
    losses = []
    # Hold accuracy results
    acc_results = np.zeros((len(test_loader.dataset), len(topk)))
    i = 0

    model.eval()
    with torch.no_grad():

        # Testing loop
        for data, targets in test_loader:

            # Tensors to gpu
            if cuda.is_available():
                data, targets = data.to('cuda'), targets.to('cuda')

            # Raw model output
            out = model(data)
            # Iterate through each example
            for pred, true in zip(out, targets):
                # Find topk accuracy
                acc_results[i, :] = accuracy(
                    pred.unsqueeze(0), true.unsqueeze(0), topk)
                classes.append(model.idx_to_class[true.item()])
                # Calculate the loss
                loss = criterion(pred.view(1, n_classes), true.view(1))
                losses.append(loss.item())
                i += 1

    # Send results to a dataframe and calculate average across classes
    results = pd.DataFrame(acc_results, columns=[f'top{i}' for i in topk])
    results['class'] = classes
    results['loss'] = losses
    results = results.groupby(classes).mean()

    return results.reset_index().rename(columns={'index': 'class'})


def add_titlebox(ax, text, nolabels, pos):
    ax.text(.55, .8, text,
    position=pos,
    transform=ax.transAxes,
    bbox=dict(facecolor='white', alpha=0.5),
    fontsize=14.5)
    ax.tick_params(axis='both', which='both', **nolabels)
    return ax


def display_prediction2(image_path, model, real_label, topk=3, save=True, show=True):
    img, ps, classes, y_obs = predict(image_path, model, topk)  # Get predictions
    result = pd.DataFrame({'p': ps}, index=classes)  # Convert results to dataframe for plotting
    win = result['p'].idxmax()
    win_prob = round(result['p'][win], 4)
    winner = win+', '+str(round(win_prob*100, 2))+'%'
    sides = ('left', 'right', 'top', 'bottom')
    nolabels = {s: False for s in sides}
    nolabels.update({'label%s' % s: False for s in sides})

    plt.figure(figsize=(8, 6))
    ax = plt.subplot(1, 1, 1)
    ax, img = imshow_tensor(img, ax=ax)
    # ax.set_title(y_obs, size=20)

    if win == real_label:
        ok_pred = True
    else:
        ok_pred = False

    winner = 'Prediction: '+winner
    add_titlebox(ax, winner, nolabels, pos=(0.03, 0.05))
    real_label = 'True Class: '+real_label
    add_titlebox(ax, real_label, nolabels, pos=(0.03, 0.11))
    plt.tight_layout()

    if save is True:
        name = win+'_'+str(round(win_prob, 2))+'_'+(y_obs.split('\\')[-1])
        plt.savefig(os.path.join('data', name))

    if show is True:
        print(f'[INFO] image: {y_obs}')  # actual image location
        print(result)
        plt.show()

    plt.close('all')

    return ok_pred, win


def process_image(image_path):
    """Process an image path into a PyTorch tensor"""

    image = Image.open(image_path)
    # Resize
    img = image.resize((256, 256))

    # Center crop
    width = 256
    height = 256
    new_width = 224
    new_height = 224

    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    img = img.crop((left, top, right, bottom))

    # Convert to numpy, transpose color dimension and normalize
    img = np.array(img).transpose((2, 0, 1)) / 256

    # Standardization
    means = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    stds = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))

    img = img - means
    img = img / stds

    img_tensor = torch.Tensor(img)

    return img_tensor


def accuracy(output, target, topk=(1, )):
    """Compute the topk accuracy(s)"""
    if cuda.is_available():
        output = output.to('cuda')
        target = target.to('cuda')

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # Find the predicted classes and transpose
        _, pred = output.topk(k=maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()

        # Determine predictions equal to the targets
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []

        # For each k, find the percentage of correct
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


def random_test_image():
    """Pick a random test image from the test directory"""
    c = np.random.choice(cat_df['category'])
    root = testdir + c + '/'
    img_path = root + np.random.choice(os.listdir(root))
    return img_path


def predict(image_path, model, topk=5):
    """Make a prediction for an image using a trained model

    Params
    --------
        image_path (str): filename of the image
        model (PyTorch model): trained model for inference
        topk (int): number of top predictions to return

    Returns

    """
    real_class = image_path  # .split('/')[-2]

    # Convert to pytorch tensor
    img_tensor = process_image(image_path)

    # Resize
    if cuda.is_available():
        img_tensor = img_tensor.view(1, 3, 224, 224).cuda()
    else:
        img_tensor = img_tensor.view(1, 3, 224, 224)

    # Set to evaluation
    with torch.no_grad():
        model.eval()
        # Model outputs log probabilities
        out = model(img_tensor)
        ps = torch.exp(out)

        # Find the topk predictions
        topk, topclass = ps.topk(topk, dim=1)

        # Extract the actual classes and probabilities
        top_classes = [
            model.idx_to_class[class_] for class_ in topclass.cpu().numpy()[0]
        ]
        top_p = topk.cpu().numpy()[0]

        return img_tensor.cpu().squeeze(), top_p, top_classes, real_class


def load_checkpoint(path):
    """Load a PyTorch model checkpoint

    Params
    --------
        path (str): saved model checkpoint. Must start with `model_name-` and end in '.pth'

    Returns
    --------
        None, save the `model` to `path`

    """

    # Get the model name
    model_name = path.split('-')[0]
    assert (model_name in ['vgg16', 'resnet50'
                           ]), "Path must have the correct model name"

    # Load in checkpoint
    checkpoint = torch.load(path)

    if model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
        # Make sure to set parameters as not trainable
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = checkpoint['classifier']

    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        # Make sure to set parameters as not trainable
        for param in model.parameters():
            param.requires_grad = False
        model.fc = checkpoint['fc']

    # Load in the state dict
    model.load_state_dict(checkpoint['state_dict'])

    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} total gradient parameters.')

    # Move to gpu
    if multi_gpu:
        model = nn.DataParallel(model)

    if cuda.is_available():
        model = model.to('cuda')

    # Model basics
    model.class_to_idx = checkpoint['class_to_idx']
    model.idx_to_class = checkpoint['idx_to_class']
    model.epochs = checkpoint['epochs']

    # Optimizer
    optimizer = checkpoint['optimizer']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer


def save_checkpoint(model, path):
    """Save a PyTorch model checkpoint

    Params
    --------
        model (PyTorch model): model to save
        path (str): location to save model. Must start with `model_name-` and end in '.pth'

    Returns
    --------
        None, save the `model` to `path`

    """

    model_name = path.split('-')[0]
    assert (model_name in ['vgg16', 'resnet50'
                           ]), "Path must have the correct model name"

    # Basic details
    checkpoint = {
        'class_to_idx': model.class_to_idx,
        'idx_to_class': model.idx_to_class,
        'epochs': model.epochs,
    }

    # Extract the final classifier and the state dictionary
    if model_name == 'vgg16':
        # Check to see if model was parallelized
        if multi_gpu:
            checkpoint['classifier'] = model.module.classifier
            checkpoint['state_dict'] = model.module.state_dict()
        else:
            checkpoint['classifier'] = model.classifier
            checkpoint['state_dict'] = model.state_dict()

    elif model_name == 'resnet50':
        if multi_gpu:
            checkpoint['fc'] = model.module.fc
            checkpoint['state_dict'] = model.module.state_dict()
        else:
            checkpoint['fc'] = model.fc
            checkpoint['state_dict'] = model.state_dict()

    # Add the optimizer
    checkpoint['optimizer'] = model.optimizer
    checkpoint['optimizer_state_dict'] = model.optimizer.state_dict()

    # Save the data to the path
    torch.save(checkpoint, path)


def train(model,
          criterion,
          optimizer,
          train_loader,
          valid_loader,
          scheduler,
          save_file_name,
          max_epochs_stop=3,
          n_epochs=20):
    """
    Params
    --------
        model (PyTorch model): cnn to train
        criterion (PyTorch loss): objective to minimize
        optimizer (PyTorch optimizier): optimizer to compute gradients of model parameters
        train_loader (PyTorch dataloader): training dataloader to iterate through
        valid_loader (PyTorch dataloader): validation dataloader used for early stopping
        save_file_name (str ending in '.pt'): file path to save the model state dict
        max_epochs_stop (int): maximum number of epochs with no improvement in validation loss for early stopping
        n_epochs (int): maximum number of training epochs
    Returns
    --------
        model (PyTorch model): trained cnn with best weights
        history (DataFrame): history of train and validation loss and accuracy
    """

    epochs_no_improve = 0  # Early stopping intialization
    valid_loss_min = np.Inf

    valid_max_acc = 0
    history = []

    # Number of epochs already trained (if using loaded in model weights)
    try:
        print(f'Model has been trained for: {model.epochs} epochs.')
    except:
        model.epochs = 0
        print(f'{bcolors.OKBLUE}STARTING TRAINING:{bcolors.ENDC}')

    overall_start = timer()

    # Main loop
    # torch.backends.cudnn.benchmark = True
    for epoch in range(n_epochs):

        train_loss = 0.0  # keep track of training and validation loss each epoch
        valid_loss = 0.0
        train_acc = 0
        valid_acc = 0

        model.train()  # Set to training
        start = timer()

        # Training loop
        for ii, (data, target) in enumerate(train_loader):
            if cuda.is_available():  # Tensors to gpu
                data, target = data.to('cuda'), target.to('cuda')
                # data, target = data.cuda(), target.cuda()

            optimizer.zero_grad()  # Clear gradients
            output = model(data)  # requires_grad, Predicted outputs are log probabilities

            loss = criterion(output, target)  # Loss and backpropagation of gradients
            loss.backward()

            optimizer.step()  # Update the parameters
            train_loss += loss.item() * data.size(0)  # Track train loss by multiplying average loss by number of examples in batch

            _, pred = torch.max(output, dim=1)  # Calculate accuracy by finding max log probability
            correct_tensor = pred.eq(target.data.view_as(pred))
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))  # Need to convert correct tensor from int to float to average
            train_acc += accuracy.item() * data.size(0)  # Multiply average accuracy times the number of examples in batch

            percent = 100 * (ii + 1) / len(train_loader)  # Track training progress
            elapsed = (timer() - start)/60
            sys.stdout.write('\rEPOCH {}: {:.1f} % complete in {:.2f} min'.format(epoch, percent, elapsed))
            sys.stdout.flush()

        # After training loops ends, start validation
        else:
            model.epochs += 1

            with torch.no_grad():  # Don't need to keep track of gradients
                model.eval()  # Set to evaluation mode

                for data, target in valid_loader:  # Validation loop
                    if cuda.is_available():  # Tensors to gpu
                        data, target = data.to('cuda'), target.to('cuda')
                        # data, target = data.cuda(), target.cuda()

                    # Forward pass
                    output = model(data)  # requires_grad=True

                    # Validation loss
                    loss = criterion(output, target)
                    # Multiply average loss times the number of examples in batch
                    valid_loss += loss.item() * data.size(0)

                    # Calculate validation accuracy
                    _, pred = torch.max(output, dim=1)
                    correct_tensor = pred.eq(target.data.view_as(pred))
                    accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
                    # Multiply average accuracy times the number of examples
                    valid_acc += accuracy.item() * data.size(0)

                train_loss = train_loss / len(train_loader.dataset)  # Calculate average losses
                valid_loss = valid_loss / len(valid_loader.dataset)
                train_acc = train_acc / len(train_loader.dataset)  # Calculate average accuracy
                valid_acc = valid_acc / len(valid_loader.dataset)

                history.append([train_loss, valid_loss, train_acc, valid_acc])

                # Print training and validation results
                print(f'\nTraining Loss: {train_loss:.4f} \t\t Training Accuracy: {100 * train_acc:.2f}%')
                print(f'Validation Loss: {valid_loss:.4f} \t Validation Accuracy: {100 * valid_acc:.2f}%\n')

                # Save the model if validation loss decreases
                if valid_loss < valid_loss_min:
                    torch.save(model.state_dict(), save_file_name)  # saving in train.py for inferences without state_dict()
                    epochs_no_improve = 0  # Track improvement
                    valid_loss_min = valid_loss
                    valid_best_acc = valid_acc
                    best_epoch = epoch
                else:  # Otherwise increment count of epochs with no improvement
                    epochs_no_improve += 1
                    if epochs_no_improve >= max_epochs_stop:  # Trigger early stopping
                        print(f'\nEarly Stopping! Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%')
                        total_time = timer() - overall_start
                        print(f'{total_time:.2f} total seconds elapsed. {total_time / (epoch+1):.2f} seconds per epoch.')

                        model.load_state_dict(torch.load(save_file_name))  # Load the best state dict
                        model.optimizer = optimizer  # Attach the optimizer

                        history = pd.DataFrame(  # Format history
                            history,
                            columns=[
                                'train_loss', 'valid_loss', 'train_acc',
                                'valid_acc'
                            ])
                        return model, history
        scheduler.step()  # the scheduler
    model.optimizer = optimizer  # Attach the optimizer
    total_time = timer() - overall_start  # Record overall time and print out stats
    print(f'\nBest epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%')
    print(f'{total_time:.2f} total seconds elapsed. {total_time / (epoch):.2f} seconds per epoch.')
    history = pd.DataFrame(history, columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc'])  # Format history
    return model, history


def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


def imshow(image):
    image = cv2.imread(image)
    #  print(image.shape) #  print(np.array(image).shape)
    image = ResizeWithAspectRatio(image, width=640)  # Resize by width only to display
    cv2.imshow('', image)
    cv2.waitKey(0)


def imshow_tensor(image, ax=None, title=None):
    """Imshow for Tensor."""

    if ax is None:
        fig, ax = plt.subplots()

    # Set the color channel as the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Reverse the preprocessing steps
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Clip the image pixel values
    image = np.clip(image, 0, 1)

    ax.imshow(image)
    # plt.axis('off')

    return ax, image


def get_pretrained_model(model_name, n_classes, multi_gpu):  # Retrieve a pre-trained model from torchvision

    if model_name == 'vgg16':
        model = models.vgg16(pretrained=True)

        for param in model.parameters():  # Freeze early layers
            param.requires_grad = False
        n_inputs = model.classifier[6].in_features

        model.classifier[6] = nn.Sequential(  # Add on classifier
            nn.BatchNorm1d(n_inputs),  # BatchNorm1d()
            nn.Linear(n_inputs, 256),
            nn.LeakyReLU(),  # nn.ReLU()
            nn.Dropout(0.4),
            nn.Linear(256, n_classes),
            nn.LogSoftmax(dim=1)
        )

    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)

        for param in model.parameters():
            param.requires_grad = False
        n_inputs = model.fc.in_features

        model.fc = nn.Sequential(
            nn.BatchNorm1d(n_inputs),  # 1d
            nn.Linear(n_inputs, 256),
            nn.BatchNorm1d(256),  # 1d
            nn.LeakyReLU(),  # nn.ReLU()
            nn.Dropout(0.2),
            nn.Linear(256, n_classes),
            nn.LogSoftmax(dim=1)
        )

    if cuda.is_available():  # Move to gpu and parallelize
        model = model.to('cuda')
        print('[INFO] Model to gpu')

    if multi_gpu:
        model = nn.DataParallel(model)

    return model


def all_plots(cat_df, history, where):

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    # fig.suptitle('METRICS', fontsize=16)

    for c in ['train_loss', 'valid_loss']:
        axs[0, 0].plot(history[c], label=c)
    axs[0, 0].legend()
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('CrossEntropy Loss')  # Average Negative Log Likelihood
    axs[0, 0].set_title('Training and Validation Losses')

    for c in ['train_acc', 'valid_acc']:
        axs[0, 1].plot(100 * history[c], label=c)
    axs[0, 1].legend()
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Average Accuracy')
    axs[0, 1].set_title('Training and Validation Accuracy')

    cat_df.plot(kind='bar', x='category', y='n_train', ax=axs[1, 0])  # panda dataframe
    cat_df.plot(kind='bar', x='category', y='n_test', color='orange', ax=axs[1, 0])
    axs[1, 0].set_ylabel('Count')
    axs[1, 0].set_xlabel('Disease')
    axs[1, 0].set_title('Training Images by Category')

    axs[1, 1].plot()

    if os.path.isfile(os.path.join(where, 'metrics.png')):
        os.remove(os.path.join(where, 'metrics.png'))
    fig.tight_layout()
    fig.savefig(os.path.join(where, 'metrics.png'))
    plt.close('all')
