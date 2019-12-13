from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from tqdm import tqdm

from core.model import Multi

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

##############################################
# Inputs
##############################################

# Top level data directory. Here we assume the format of the directory conforms
#   to the ImageFolder structure
jupyter = torch.cuda.is_available()

if jupyter:
    data_dir  = "/content/drive/My Drive/dataset/dataset"
else:
    data_dir = "/media/stephane/DATA/ESILV/A5/IA for IOT/Projet Final/hymenoptera_data/"
    # data_dir = "/media/stephane/DATA/ESILV/A5/IA for IOT/Projet Final/dataset/"


# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "squeezenet"

# Number of classes in the dataset
num_classes = 17

# Batch size for training (change depending on how much memory you have)
batch_size = 32

# Number of epochs to train for
num_epochs = 20

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = True

# Yes we'll always use pretrained because we need to gain some time
use_pretrained = True

noise = 1e-2


##############################################
# Initialize and Reshape the Networks (MAIN)
##############################################

# Initialize the model for this run
multi = Multi(model_name, num_classes, feature_extract, use_pretrained, lr=0.001, momentum=0.9)
multi_label= "Squeezenet LR 0.001 Momentum 0.9"

##############################################
# Initialize and Reshape the Networks (COMPARED TO SCRATCH OR ELSE)
##############################################
model_name2 = "resnet"
multi_scratch = Multi(model_name2, num_classes, feature_extract, use_pretrained, lr=0.001, momentum=0.9)
multi_scratch_label= "resnet LR 0.001 Momentum 0.9"


##############################################
# Model Training and Validation Code
##############################################
def train_model(multi, dataloaders, criterion, num_epochs=25):
    
    model = multi.model_ft
    optimizer = multi.optimizer_ft
    device = multi.device
    is_inception = multi.model_name == "inception"
    

    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = multi.forward(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = multi.forward(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history




##############################################
# Load Data
##############################################

# Data augmentation and normalization for training
# Just normalization for validation
def create_data_loaders( input_size):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation( degrees = 360 ),
            transforms.Resize( size = input_size ),
            transforms.RandomCrop( size = input_size ),
            transforms.RandomVerticalFlip( p = 0.5 ),
            transforms.ColorJitter( brightness = .2, contrast = .2, saturation = .2, hue = .1 ),
            transforms.ToTensor( ),
            transforms.Lambda( lambda X: X * ( 1. - noise ) + torch.randn( X.shape ) * noise ),
            transforms.Normalize( [ 0.485, 0.456, 0.406 ], [ 0.229, 0.224, 0.225 ] )
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

    return dataloaders_dict





##############################################
# Run Training and Validation Step
##############################################


# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

# Train and evaluate
model_ft, hist = train_model(multi, create_data_loaders( multi.input_size ), criterion, num_epochs)


##############################################
# Comparison with Model Trained from Scratch
##############################################

# Run scratch or other compared model

scratch_criterion = nn.CrossEntropyLoss()
_,scratch_hist = train_model(multi_scratch,  create_data_loaders( multi_scratch.input_size ), scratch_criterion, num_epochs)

# Plot the training curves of validation accuracy vs. number
#  of training epochs for the transfer learning method and
#  the model trained from scratch
ohist = []
shist = []

ohist = [h.cpu().numpy() for h in hist]
shist = [h.cpu().numpy() for h in scratch_hist]

#print(multi.traced_model_ft.code)

plt.title("Validation Accuracy vs. Number of Training Epochs")
plt.xlabel("Training Epochs")
plt.ylabel("Validation Accuracy")
plt.plot(range(1,num_epochs+1),ohist,label=multi_label)
plt.plot(range(1,num_epochs+1),shist,label=multi_scratch_label)
plt.ylim((0,1.))
plt.xticks(np.arange(1, num_epochs+1, 1.0))
plt.legend()
plt.show()


