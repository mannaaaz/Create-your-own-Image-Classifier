import numpy as np
import argparse
import torch
from torch import nn, optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.autograd import Variable
from collections import OrderedDict
import matplotlib.pyplot as plt
import json
from PIL import Image
import futility
import fmodel

parser = argparse.ArgumentParser (description = 'Training Script')

parser.add_argument ('data_dir', help = 'Input data directory. Mandatory', type=str)
parser.add_argument ('--save_dir', help = 'Input saving directory. Optional',  type=str)
parser.add_argument ('--arch', help = 'Default is Alexnet, otherwise input VGG13', type=str, default='alexnet')
parser.add_argument ('--learning_r', help = 'Learning rate - default is 0.001', type = float, default = 0.001)
parser.add_argument ('--hidden_units', help = 'Hidden units. Default val 2048', type = int, default = 2048)
parser.add_argument ('--epochs', help = 'Epochs as integer - default is 5', type = int, default = 5)
parser.add_argument ('--GPU', help = "Input GPU if you want to use it", type = str)



    args = parser.parse_args()

    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms =   transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomResizedCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])


# TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir,                        transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir,                       transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir,                         transform=test_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloaders = torch.utils.data.DataLoader(train_data,             batch_size=64, shuffle=True)
    validloaders = torch.utils.data.DataLoader(valid_data,              batch_size=64, shuffle=True)
    testloaders = torch.utils.data.DataLoader(test_data,                batch_size=64, shuffle=True)

    with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# BUILDING AND TRAINING CLASSIFIER

if args.arch == 'vgg13':
    model = models.vgg13(pretrained=True)
    no_input_layer = 25088
else:
    model = models.alexnet(pretrained=True)
    no_input_layer = 9216

for param in model.parameters():
    param.requires_grad = False

if args.hidden_units != None:
    classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(no_input_layer, args.hidden_units)),
                            ('relu1', nn.ReLU()),
                            ('dropout1', nn.Dropout(p=0.3)),
                            ('fc2', nn.Linear(args.hidden_units, 2048)),
                            ('relu2', nn.ReLU()),
                            ('dropout2', nn.Dropout(p=0.3)),
                            ('fc3', nn.Linear(2048, 102)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))
    
else:
    classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(no_input_layer, 4096)),
                            ('relu1', nn.ReLU()),
                            ('dropout1', nn.Dropout(p=0.3)),
                            ('fc2', nn.Linear(4096, 2048)),
                            ('relu2', nn.ReLU()),
                            ('dropout2', nn.Dropout(p=0.3)),
                            ('fc3', nn.Linear(2048, 102)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))

model.classifier = classifier

# initializing criterion and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), args.learning_r)

if args.GPU == 'GPU':
    device = 'cuda'
    print("You chose to enable the GPU")
else:
    device = 'cpu'
   
# defining validation
def validation(model, valid_loader, criterion):
    model.to(device)

    valid_loss = 0
    accuracy = 0
    for inputs, labels in valid_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        output = model.forward(inputs)
        valid_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return valid_loss, accuracy


# training the model
model.to(device)
epochs = args.epochs
print_every = 15
steps = 0

print(f"\nINITIALIZING TRAINING PHASE WITH {args.epochs} EPOCHS")
print(f"Training {args.arch} network architecture.")
print(f"Learning rate is set to {args.learning_r}.")

for e in range(epochs):
    running_loss = 0
    for ii, (inputs, labels) in enumerate(train_loader):
        steps += 1

        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if steps % print_every == 0:
            model.eval()

            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                valid_loss, accuracy = validation(model, valid_loader, criterion)

            print('Epoch: {}/{}.. '.format(e + 1, epochs),
                  'Training Loss: {:.2f}.. '.format(
                      running_loss / print_every),
                  'Valid Loss: {:.2f}.. '.format(
                      valid_loss / len(valid_loader)),
                  'Valid Accuracy: {:.2f}%'.format(accuracy / len(valid_loader) * 100))

            running_loss = 0

            model.train()

# saving to checkpoint
model.class_to_idx = train_image_datasets.class_to_idx
checkpoint = {'classifier': model.classifier, 'state_dict': model.state_dict (), 'class_to_idx': model.class_to_idx,'arch': args.arch} 

if args.save_dir:
    torch.save (checkpoint, args.save_dir + '/checkpoint.pth')
else:
    torch.save (checkpoint, 'checkpoint.pth')          
        