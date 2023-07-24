import torch
from torch import nn
from torchvision import models

def create_model(arch='vgg13', hidden_units=512, num_classes=102):
    if arch == 'vgg13':
        model = models.vgg13(pretrained=True)
        input_size = model.classifier[0].in_features
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        input_size = model.classifier.in_features
    else:
        raise ValueError("Unsupported architecture. Please choose either 'vgg13' or 'densenet121'.")

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(
        nn.Linear(input_size, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_units, num_classes),
        nn.LogSoftmax(dim=1)
    )

    model.classifier = classifier
    return model

def save_checkpoint(model, optimizer, arch, hidden_units, save_dir):
    checkpoint = {
        'arch': arch,
        'hidden_units': hidden_units,
        'state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_to_idx': model.class_to_idx
    }
    torch.save(checkpoint, save_dir)

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = create_model(checkpoint['arch'], checkpoint['hidden_units'])
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model
