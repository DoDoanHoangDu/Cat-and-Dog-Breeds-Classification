import torch
from torch import nn
import pandas as pd
from torch import optim
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import random
import time
from tqdm import tqdm
import copy
import os
from torchvision import datasets
from torch.utils.data import ConcatDataset, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()
#parameter declaration


leaning_rate = 1e-4
num_classes = 120
patch_size = 4
img_size = 150
in_channels = 3
num_heads = 6
dropout = 0.001
hidden_dim = 22500
adam_weight_decay = 0
adam_betas = (0.9,0.999)
activation = "gelu"
num_encoders = 3
embed_dim = (patch_size ** 2) * in_channels
num_patches = (img_size//patch_size) ** 2

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Using {device}.')
if torch.backends.cuda.flash_sdp_enabled():
    print("Flash SDP is enabled")
else:
    print("Flash SDP is disabled")


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# patch embedding

class PatchEmbedding(nn.Module):
    def __init__(self, embed_dim, patch_size, num_patches, dropout, in_channels):
        super().__init__()
        # using cnn to split image into patches, then applied convolution to embed the patches to embed_size
        self.patcher = nn.Sequential(
            nn.Conv2d(
                in_channels= in_channels,
                out_channels= embed_dim,
                kernel_size = patch_size,
                stride = patch_size
            ),
            #Input shape: (batch_size, channels, height, width)
            #Output shape: (batch_size, channels, height * width)
            nn.Flatten(2))

        #learnable classification token, append to the patch embeddings. This capturing the global information
        #for classification, similarity to [CLS] token in BERT
        self.cls_token = nn.Parameter(torch.randn(size = (1,1,embed_dim) ),requires_grad = True)
        
        # if want to train images with multiple channels, change as follow:
        # self.cls_token = nn.Parameter(torch.randn(size=(1, in_channels, embed_dim)), requires_grad=True) 
        # to self.cls_token = nn.Parameter(torch.randn(size=(1, 1, embed_dim)), requires_grad=True).
        
        #adding the 
        self.position_embeddings = nn.Parameter(torch.randn(size = (1,num_patches + 1,embed_dim) ),requires_grad = True)
        self.dropout = nn.Dropout(p = dropout)
    def forward(self,x):
        cls_token = self.cls_token.expand(x.shape[0],-1,-1)

        x = self.patcher(x).permute(0,2,1)
        x = torch.cat([cls_token,x],dim=1) #class token is concatenated with the patches
        
        #adding position information
        x = self.position_embeddings + x

        x = self.dropout(x)
        return x





class VisionTransformer(nn.Module):
    def __init__(self, num_patches, num_classes, patch_size, embed_dim , num_encoders,
                num_heads,hidden_dim, dropout, activation, in_channels):
        super().__init__()
        self.embeddings_block = PatchEmbedding(embed_dim,patch_size,num_patches,dropout,in_channels)

        encoder_layer = nn.TransformerEncoderLayer(d_model= embed_dim,nhead=num_heads,dropout=dropout,
                                                    activation=activation, batch_first= True, norm_first= False)
        
        
        #stack of transformer encoder layers
        self.encoder_blocks = nn.TransformerEncoder(encoder_layer, num_layers= num_encoders)
        
        #FLC layers to perform the final classification
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(normalized_shape = embed_dim),
            nn.Linear(in_features = embed_dim, out_features = num_classes),
        )
    def forward(self,x):
        x = self.embeddings_block(x)

        x = self.encoder_blocks(x)
        
        x = self.mlp_head(x[:, 0, :])

        return x

vision_transformer = VisionTransformer(num_patches, num_classes, patch_size, embed_dim, num_encoders,
                                        num_heads, hidden_dim, dropout,activation,in_channels)
vision_transformer.to(device)
scripted_model = torch.jit.script(vision_transformer)

data_processing_time = time.time()

#read the dataset from folder

data_dir = r'C:\Users\dmin\HUST\20241\DeepLearning\Images'

# Define data transforms
base_transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor()
])

rotate_90_transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.RandomRotation(degrees=(90, 90)),
    transforms.ToTensor()
])

rotate_180_transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.RandomRotation(degrees=(180, 180)),
    transforms.ToTensor()
])

random_transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomResizedCrop(150, scale=(0.8, 1.0)),
    transforms.ToTensor(),
])

#Load the datasets
original_dataset = datasets.ImageFolder(root=data_dir, transform=base_transform)
rotated_90_dataset = datasets.ImageFolder(root=data_dir, transform=rotate_90_transform)
rotated_180_dataset = datasets.ImageFolder(root=data_dir, transform=rotate_180_transform)
random_dataset = datasets.ImageFolder(root=data_dir, transform=random_transform)


combined_dataset = ConcatDataset([original_dataset, rotated_90_dataset, rotated_180_dataset, random_dataset])


targets = np.array([label for _, label in combined_dataset])


stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)


for train_idx, val_idx in stratified_split.split(np.zeros(len(targets)), targets):
    train_dataset = Subset(combined_dataset, train_idx)
    val_dataset = Subset(combined_dataset, val_idx)


train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

print(f"Data processing phase: {(time.time() - data_processing_time):.2f}s")
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")



#training process
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vision_transformer.parameters(), betas = adam_betas,lr= leaning_rate, weight_decay= adam_weight_decay)


def test_the_model(model, test_dataloader):
    model.eval()  
    correct = 0
    total = 0
    running_loss = 0.0

    
    with torch.no_grad():
        for images, labels in test_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            probabilities = torch.softmax(outputs, dim=1)
            
            loss = criterion(outputs, labels)

            # Get the top k predicted labels for each sample
            _, topk_preds = torch.topk(probabilities, k=1, dim=1)

            # Check if the true label is in the top 5 predicted labels
            correct += torch.sum(topk_preds.eq(labels.view(-1, 1))).item()

            total += labels.size(0)
            running_loss += loss.item()

    # Calculate test accuracy based on top-k predictions
    accuracy = (correct / total) * 100
    avg_loss = running_loss / len(test_dataloader)
    return accuracy, avg_loss


def train_the_model(num_epochs=5):
    print("Starting training process...")
    accuracies = []
    test_accuracies = []
    max_accuracy = 0
    best_model = None

    for epoch in range(num_epochs):
        vision_transformer.train()
        correct = 0
        total = 0
        running_loss = 0.0
        start_epoch = time.time()

        for i, (images, labels) in enumerate(train_dataloader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            
            with autocast():
                outputs = vision_transformer(images)
                loss = criterion(outputs, labels)

            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            running_loss += loss.item()

        time_complete_epoch = time.time() - start_epoch
        train_accuracy = (correct / total) * 100
        accuracies.append(train_accuracy)

        test_accuracy, test_loss = test_the_model(vision_transformer, val_dataloader)
        test_accuracies.append(test_accuracy)

        if test_accuracy > max_accuracy:
            best_model = copy.deepcopy(vision_transformer)
            max_accuracy = test_accuracy
            print(f"Saving best model with Test Accuracy: {test_accuracy:.2f}%")
        
        print(f"Epoch {epoch + 1}/{num_epochs}, "
              f"Train Loss: {running_loss / len(train_dataloader):.4f}, "
              f"Train Accuracy: {train_accuracy:.2f}%, "
              f"Test Loss: {test_loss:.4f}, "
              f"Test Accuracy: {test_accuracy:.2f}%, "
              f"Time: {time_complete_epoch:.2f} seconds")

    plt.plot(accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Testing Accuracy Over Epochs")
    plt.legend()
    plt.show()

    return best_model

if __name__ == '__main__':
    print("Starting training...")
    start_training_time = time.time()
    result_model = train_the_model(60)
    print(f'Training time: {(time.time() - start_training_time) / 60} minutes.')
