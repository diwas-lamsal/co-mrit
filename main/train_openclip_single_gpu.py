import torch
from torch.utils.data import DataLoader
from torch import nn
from tab_transformer_pytorch import FTTransformer
from models.CoMRIT import CoMRIT
from models.vision_transformer3d import ViT
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from datasets import CombinedDataset

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

from utils.prepare_data import prepare_combined_dataset


# Seed for reproducibility
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_image_paths, X_train_cat, X_train_cont, y_train, cat_mask_train, num_mask_train, test_image_paths, X_test_cat, X_test_cont, \
y_test, cat_mask_test, num_mask_test, num_categories, num_continuous, categorical_columns, continuous_columns = prepare_combined_dataset()

# Define dimensions for output embedding and projection vectors
image_feature_dim = 2048
tabular_feature_dim = 2048
projection_dim = 2048
num_classes = 5

# Whether to run parallel classification
parallel_classify = True

# Define data dimensions for the MRI images 
image_height = 96
image_depth = 96
image_width = 96
image_channels = 1  # Assuming grayscale MRI images

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
train_dataset = CombinedDataset(
    image_paths=train_image_paths,
    cat_features=X_train_cat,
    cont_features=X_train_cont,
    labels=y_train,
    cat_masks=cat_mask_train,
    num_masks=num_mask_train,
    transforms=data_transforms,
    path_prefix = "/home/ssd/"
)

test_dataset = CombinedDataset(
    image_paths=test_image_paths,
    cat_features=X_test_cat,
    cont_features=X_test_cont,
    labels=y_test,
    cat_masks=cat_mask_test,
    num_masks=num_mask_test,
    transforms=data_transforms,
    path_prefix = "/home/ssd/"
)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print(sum(num_categories))

tabular_encoder = FTTransformer(
    categories=num_categories,
    num_continuous=len(continuous_columns),
    dim=32,
    dim_out=num_classes, 
    depth=6,
    heads=8,
    attn_dropout = 0.1,                 # post-attention dropout
    ff_dropout = 0.1                    # feed forward dropout
).to(device)

load_tabular_weights = True
if load_tabular_weights:
    tabular_encoder.load_state_dict(torch.load("pretrained/epoch_2_NACC_ft_transformer.pth"),strict=False)
    tabular_encoder.dim_out = tabular_feature_dim

image_encoder = ViT(
    image_size = (image_height,image_width),          # image size
    depth = image_depth,               # number of frames
    image_patch_size = 16,     # image patch size
    depth_patch_size = 16,      # frame patch size
    num_classes = image_feature_dim,
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

load_imaging_weights = True
if load_imaging_weights:
    image_encoder.load_state_dict(torch.load("pretrained/best_model_490_patchsz16.pth"),strict=False)


contrastive_model = CoMRIT(
    image_encoder=image_encoder,
    tabular_encoder=tabular_encoder,
    image_feature_dim = image_feature_dim, 
    tabular_feature_dim = tabular_feature_dim, 
    projection_dim = projection_dim, 
    parallel_classify=parallel_classify,
    num_classes=num_classes,
).to(device)


optimizer = torch.optim.AdamW(contrastive_model.parameters(), lr=0.0001, weight_decay=0.00001)
best_loss = float('inf')


def train_epoch(model, train_loader, optimizer, device):
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0
    running_loss_contrastive = 0.0
    running_loss_classification = 0.0
    pbar = tqdm(train_loader, desc='Training Loss: 0.0000')

    for i, (images, cat_features, cont_features, y_true, cat_mask, num_mask) in enumerate(pbar):
        images, cat_features, cont_features, y_true = images.to(device), cat_features.to(device), cont_features.to(device), y_true.to(device)
        cat_mask = cat_mask.to(device)
        num_mask = num_mask.to(device)
        
        optimizer.zero_grad()
        outputs = model(images,cat_features,cont_features,y_true) 
        
        loss = outputs['contrastive_loss']
        running_loss_contrastive += outputs["contrastive_loss"].item()
        
        if 'classification_loss' in outputs:
            loss += outputs['classification_loss']
            _, predicted = outputs['classification_logits'].max(1)
            correct += predicted.eq(y_true).sum().item()
            total += y_true.size(0)
            running_loss_classification += outputs["classification_loss"].item()
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        pbar.set_description(f'Training Loss: {running_loss/(i+1):.4f}')

    average_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total if total > 0 else 0
    
    average_contrastive_loss = running_loss_contrastive/len(train_loader)
    average_classification_loss = running_loss_classification/len(train_loader)
    
    return average_loss, accuracy, average_contrastive_loss, average_classification_loss


def valid_epoch(model, valid_loader, device):
    model.eval()
    running_loss = 0.0
    total = 0
    correct = 0
    running_loss_contrastive = 0.0
    running_loss_classification = 0.0
    pbar = tqdm(valid_loader, desc='Validation Loss: 0.0000')

    with torch.no_grad():
        for i, (images, cat_features, cont_features, y_true, cat_mask, num_mask) in enumerate(pbar):
            images, cat_features, cont_features, y_true = images.to(device), cat_features.to(device), cont_features.to(device), y_true.to(device)
            cat_mask = cat_mask.to(device)
            num_mask = num_mask.to(device)

            outputs = model(images, cat_features, cont_features, y_true)
            loss = outputs['contrastive_loss']
            running_loss_contrastive += outputs["contrastive_loss"].item()
            
            if 'classification_loss' in outputs:
                loss += outputs['classification_loss']
                _, predicted = outputs['classification_logits'].max(1)
                correct += predicted.eq(y_true).sum().item()
                total += y_true.size(0)
                running_loss_classification += outputs["classification_loss"].item()
                
            running_loss += loss.item()
            pbar.set_description(f'Validation Loss: {running_loss/(i+1):.4f}')

    average_loss = running_loss / len(valid_loader)
    accuracy = 100. * correct / total if total > 0 else 0
    
    average_contrastive_loss = running_loss_contrastive/len(valid_loader)
    average_classification_loss = running_loss_classification/len(valid_loader)
    
    return average_loss, accuracy, average_contrastive_loss, average_classification_loss


def setup_scheduler(optimizer, scheduler_type, total_epochs, switch_epoch=None):
    if scheduler_type == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=0)  # Complete run with cosine decay
    elif scheduler_type == 'step':
        assert switch_epoch is not None, "Switch epoch must be defined for step scheduler"
        # Set the step scheduler to activate at 80% of total epochs
        return StepLR(optimizer, step_size=switch_epoch, gamma=0.1)
    else:
        return None  # No scheduler used


# Configuration for scheduler
scheduler_type = 'cosine'  # Options: 'cosine', 'step', or None
total_epochs = 100
switch_epoch = int(0.8 * total_epochs) if scheduler_type == 'step' else None

# Setup the scheduler
scheduler = setup_scheduler(optimizer, scheduler_type, total_epochs, switch_epoch)

for epoch in range(100):
    train_loss, train_acc, train_cont_loss, train_class_loss = train_epoch(contrastive_model, train_dataloader, optimizer, device)
    valid_loss, valid_acc, val_cont_loss, val_class_loss = valid_epoch(contrastive_model, test_dataloader, device)
    if parallel_classify:
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Contrastive Loss: {train_cont_loss}, Train Classification Loss: {train_class_loss}, " 
            + f"Valid Loss: {valid_loss:.4f}, Val Contrastive Loss: {val_cont_loss}, Val Classification Loss: {val_class_loss}, " 
            + f" Train Acc:{train_acc}, Valid_acc:{valid_acc}")
    else:
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")
    if valid_loss < best_loss:
        best_loss = valid_loss
        torch.save(contrastive_model.state_dict(), 'best_model.pth')
        print("Saved Best Model!")

    # Update the scheduler if one is used
    if scheduler:
        scheduler.step()

