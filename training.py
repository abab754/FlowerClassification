import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import os

def train_model():
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

    # Load pre-trained ResNet model
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Modify the final layer to match the number of classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 102)  # 102 classes in the flowers dataset

    # Move the model to the GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Training loop
    for epoch in range(10):  # Loop over the dataset multiple times
        print(f"Epoch {epoch+1}/10")
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            
            running_loss += loss.item()
            
            # Calculate accuracy
            _, preds = torch.max(outputs, 1)
            correct_predictions += (preds == labels).sum().item()
            total_predictions += labels.size(0)

        epoch_loss = running_loss / len(dataloaders['train'])
        epoch_accuracy = correct_predictions / total_predictions

        print(f"Training Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct_predictions = 0
        val_total_predictions = 0

        with torch.no_grad():
            for inputs, labels in dataloaders['val']:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, preds = torch.max(outputs, 1)
                val_correct_predictions += (preds == labels).sum().item()
                val_total_predictions += labels.size(0)

        val_loss /= len(dataloaders['val'])
        val_accuracy = val_correct_predictions / val_total_predictions

        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

        # Step the scheduler
        scheduler.step()

    print("Finished Training")

    # Save the trained model
    torch.save(model.state_dict(), 'resnet50_flowers.pth')
    print("Model saved as resnet50_flowers.pth")
    
    return model, device, criterion

def test_model(model, device, criterion):
    # Define test data transformations
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load the test dataset
    test_dir = 'flowers/test'
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Test the model
    model.eval()
    test_running_loss = 0.0
    test_correct_predictions = 0
    test_total_predictions = 0

    with torch.no_grad():  # Disable gradient calculation
        for test_inputs, test_labels in test_loader:
            test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)

            test_outputs = model(test_inputs)
            test_loss = criterion(test_outputs, test_labels)
            test_running_loss += test_loss.item()

            _, test_preds = torch.max(test_outputs, 1)
            test_correct_predictions += (test_preds == test_labels).sum().item()
            test_total_predictions += test_labels.size(0)

    test_epoch_loss = test_running_loss / len(test_loader)
    test_epoch_accuracy = test_correct_predictions / test_total_predictions

    print(f"Test Loss: {test_epoch_loss:.4f}, Test Accuracy: {test_epoch_accuracy:.4f}")

if __name__ == '__main__':
    model, device, criterion = train_model()  # Train the model
    test_model(model, device, criterion)  # Test the model on the test dataset