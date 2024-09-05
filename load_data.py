import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the transformations for the dataset
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'flowers'
image_datasets = {
    'train': datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=data_transforms['train']),
    'val': datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=data_transforms['val']),
}
dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True, num_workers=4),
    'val': DataLoader(image_datasets['val'], batch_size=32, shuffle=True, num_workers=4),
}

# Print the number of images in each dataset
print(f"Number of training images: {len(image_datasets['train'])}")
print(f"Number of validation images: {len(image_datasets['val'])}")

data_dir = 'flowers'
image_datasets = {
    'train': datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=data_transforms['train']),
    'val': datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=data_transforms['val']),
}
dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True, num_workers=4),
    'val': DataLoader(image_datasets['val'], batch_size=32, shuffle=True, num_workers=4),
}

# Print the number of images in each dataset
print(f"Number of training images: {len(image_datasets['train'])}")
print(f"Number of validation images: {len(image_datasets['val'])}")
