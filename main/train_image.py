import torch
from torch.utils.data import DataLoader
from torch import nn
from utils.prepare_data import prepare_image_dataset
from datasets import MRIDataset
from models.vision_transformer3d import ViT
from tqdm import tqdm

from monai.transforms import (
    Compose,
    LoadImaged,
    ToTensord,
    EnsureChannelFirstd,
    DivisiblePadd,
    Resized,
    HistogramNormalized,
    NormalizeIntensityd
)

# Seed for reproducibility
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_image_paths, y_train, test_image_paths, y_test = prepare_image_dataset()


# Define data dimensions for the MRI images 
image_height = 96
image_depth = 80
image_width = 80
image_channels = 1  # Assuming grayscale MRI images

# Original:
# image_height = 192
# image_width = 160
# image_depth = 160

keys=['path']
data_transforms = Compose(
    [
        LoadImaged(keys, reader='NibabelReader'),
        EnsureChannelFirstd(keys),
        DivisiblePadd(keys, k=16),
        Resized(keys, spatial_size=(image_height, image_depth, image_width)),
        NormalizeIntensityd(keys, subtrahend=0, divisor=255),
        ToTensord(keys, dtype=torch.float32)
    ]
)

# Create dataset instances
train_dataset = MRIDataset(
    image_paths=train_image_paths,
    labels=y_train,
    transforms=data_transforms,
    path_prefix = "/home/ssd/"
)

# Create dataset instances
test_dataset = MRIDataset(
    image_paths=test_image_paths,
    labels=y_test,
    transforms=data_transforms,
    path_prefix = "/home/ssd/"
)

# DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)

num_classes = 5
model = ViT(
    image_size = (image_height,image_width),          # image size
    depth = image_depth,               # number of frames
    image_patch_size = 16,     # image patch size
    depth_patch_size = 16,      # frame patch size
    num_classes = num_classes,
    channels=1,
    dim = 768,
    layers = 12,
    heads = 12,
    pool = 'cls',
    dim_head= 64,
    mlp_dim = 3072,
    dropout = 0.1,
    emb_dropout = 0.0
).to(device)

def multi_class_accuracy(y_pred, y_true):
    # Get the class predictions from the maximum value of the logits
    _, predicted = torch.max(y_pred, 1)
    correct = (predicted == y_true).float()  # Compare with true labels
    acc = correct.sum() / len(correct)  # Calculate accuracy
    return acc

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
num_epochs = 100

for _, epoch in enumerate(range(num_epochs)):
    model.train()
    total_loss = 0
    total_acc = 0
    count = 0

    # Wrap the dataloader with tqdm for the training loop
    train_iterator = tqdm(train_dataloader, desc=f'Epoch {epoch}/{num_epochs-1} - Train')
    for images, labels in train_iterator:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels.squeeze())

        # Backward and optimize
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        acc = multi_class_accuracy(outputs, labels.squeeze())
        total_loss += loss.item()
        total_acc += acc.item()
        count += 1

        # Update tqdm description with current loss and accuracy
        train_iterator.set_description(f'Epoch {epoch}/{num_epochs-1} - Train Loss: {total_loss/count:.4f}, Acc: {total_acc/count:.4f}')

    average_loss = total_loss / count
    average_acc = total_acc / count
    
    if epoch==num_epochs-1:
        print(f'End of Epoch {epoch}, Train Loss: {average_loss:.4f}, Train Accuracy: {average_acc:.4f}')
    
    # Validation loop every 2 epochs
    if epoch % 2 == 0 or epoch==num_epochs-1:
        model.eval()
        total_val_loss = 0
        total_val_acc = 0
        val_count = 0
        
        # Wrap the dataloader with tqdm for the validation loop
        val_iterator = tqdm(test_dataloader, desc=f'Epoch {epoch}/{num_epochs-1} - Validate')
        with torch.no_grad():
            for images, labels in val_iterator:
                images, labels = images.to(device), labels.to(device)
                
                # Forward pass
                val_outputs = model(images)
                val_loss = criterion(val_outputs, labels.squeeze())
                
                val_acc = multi_class_accuracy(val_outputs, labels.squeeze())
                total_val_loss += val_loss.item()
                total_val_acc += val_acc.item()
                val_count += 1

                # Update tqdm description with current validation loss and accuracy
                val_iterator.set_description(f'Epoch {epoch}/{num_epochs-1} - Val Loss: {total_val_loss/val_count:.4f}, Acc: {total_val_acc/val_count:.4f}')
        
        average_val_loss = total_val_loss / val_count
        average_val_acc = total_val_acc / val_count
        
        if epoch==num_epochs-1:
            print(f'End of Epoch {epoch}, Val Loss: {average_val_loss:.4f}, Val Accuracy: {average_val_acc:.4f}')
        

