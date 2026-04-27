import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm

from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from tqdm import tqdm

class CustomDataset(Dataset):
    # initialization
    # transform is applied to the data
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)

    # return length of dataset   
    def __len__(self):
        return len(self.data)

    # return one item on the index
    def __getitem__(self, idx):
        return self.data[idx]
    
    # return the data classes from folder
    @property
    def classes(self):
        return self.data.classes
    
class SimpleCNN(nn.Module):
    # define all parts of model
    def __init__(self, num_classes = 6):
        super(SimpleCNN, self).__init__()
        self.base_mode = timm.create_model('efficientnet_b0', pretrained=True)
        self.features = nn.Sequential(*list(self.base_mode.children())[:-1])

        enet_out_size = 1280
        # create classifier
        self.classifier = nn.Linear(enet_out_size, num_classes)

    # connect all parts of model together    
    def forward(self, x):   
        x = self.features(x)
        output = self.classifier(x)
        return output

# load and preprocess the image
def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    return image, transform(image).unsqueeze(0)

# predict using the model
def predict(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return probabilities.cpu().numpy().flatten()

# visualization
def visualize_predictions(original_image, probabilities, class_names):
    fig, axarr = plt.subplots(1, 2, figsize=(14, 7))
    
    # Display image
    axarr[0].imshow(original_image)
    axarr[0].axis("off")
    
    # Display predictions
    axarr[1].barh(class_names, probabilities)
    axarr[1].set_xlabel("Probability")
    axarr[1].set_title("Class Predictions")
    axarr[1].set_xlim(0, 1)

    plt.tight_layout()
    plt.show()

def main():
    # model expects all images in same size
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = CustomDataset(data_dir='\\Wildlife_Detection_CNN\\classify_dataset\\train', transform=transform)
    
    # Dataloader
    # Batchsize is the number of samples to pull from the dataset at a time
    # Shuffle is to randomize the order of the data, only needed for training
    dataLoader = DataLoader(dataset, batch_size =32 , shuffle=True)

    # for images, labels in dataLoader:
    #     print(f"Batch of images shape: {images.shape}")
    #     print(f"Batch of labels shape: {labels.shape}")
    #     print(f"Batch of labels: {labels}")
    #     break

    # print(f"Dataset length: {dataset.__len__()}")
    # print(f"Dataset classes: {dataset.classes}")
    # print(f"First item in dataset: {dataset.__getitem__(0)}")

    # image, label = dataset.__getitem__(0)
    # print(f"Label: {label}")
    # print(f"Class: {target_to_class.get(label)}")

    # target_to_class = {v: k for k, v in ImageFolder(data_dir).class_to_idx.items()}
    # print(f"Target to class mapping: {target_to_class}")

    for images, labels in dataLoader:
        break

    model = SimpleCNN(num_classes=6)

    example = model(images)
    # print(example)
    # # shape is bath size x number of classes
    # print(example.shape)

    train_folder = '\\Wildlife_Detection_CNN\\classify_dataset\\train'
    val_folder = '\\Wildlife_Detection_CNN\\classify_dataset\\val'
    test_folder = '\\Wildlife_Detection_CNN\\classify_dataset\\test'

    train_dataset = CustomDataset(data_dir=train_folder, transform=transform)
    val_dataset = CustomDataset(data_dir=val_folder, transform=transform)   
    test_dataset = CustomDataset(data_dir=test_folder, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Training loop
    # pick a loss function and an optimizer
    criterion = nn.CrossEntropyLoss()
    optimzer = optim.Adam(model.parameters(), lr=0.001)

    # run on gpu
    print(f"cuda available: {torch.cuda.is_available()}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    # move model to gpu
    model.to(device)

    # number of times to loop through the entire dataset
    num_epochs = 5

    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        # keep track of loss for each epoch
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimzer.zero_grad()

            # run the model on the images and record loss
            outputs = model(images)
            loss = criterion(outputs, labels)

            # back propagate the loss and update the model parameters
            loss.backward()
            optimzer.step()
            running_loss += loss.item() * images.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # validation
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Validation"):
                # same thing but on the validation set, no back propagation or optimizer step
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)
        val_loss =  running_loss / len(val_loader.dataset)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # plot the training loss
    plt.plot(train_losses, label = "training losses")
    plt.plot(val_losses, label = "validation losses")
    plt.legend()
    plt.title("losses over epoch")
    plt.show()

    # evaulate model on test image
    test_image = "\\Wildlife_Detection_CNN\\classify_dataset\\test\\4\\001265.png"
    original_image, image_tensor = preprocess_image(test_image, transform)
    probabilities = predict(model, image_tensor, device)
    class_names = dataset.classes 
    visualize_predictions(original_image, probabilities, class_names)

if __name__ == "__main__":
    main()