import json
import numpy as np
from PIL import Image
import torch
from torchvision import transforms, datasets

def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define transforms for the training, validation, and testing sets
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # Load the datasets with ImageFolder
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    }

    # Define the dataloaders
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=64)
    }

    return dataloaders['train'], dataloaders['valid'], dataloaders['test']

def process_image(image_path):
    with Image.open(image_path) as img:
        # Resize the image while maintaining aspect ratio
        img.thumbnail((256, 256))
        # Center crop the image to 224x224 pixels
        left = (img.width - 224) / 2
        top = (img.height - 224) / 2
        right = left + 224
        bottom = top + 224
        img = img.crop((left, top, right, bottom))
        # Convert image to numpy array
        np_image = np.array(img) / 255
        # Normalize the image
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        np_image = (np_image - mean) / std
        # Reorder dimensions for PyTorch
        np_image = np_image.transpose((2, 0, 1))
        # Convert numpy array to tensor
        image_tensor = torch.from_numpy(np_image).type(torch.FloatTensor)
        return image_tensor

def load_categories(category_names):
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name
