import torch
from tab_transformer_pytorch import TabTransformer, FTTransformer
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
from utils.prepare_data import prepare_tabular_dataset
from tqdm import tqdm

# Seed for reproducibility
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train_cat, X_train_cont, y_train, cat_mask_train, num_mask_train, X_test_cat, X_test_cont, y_test, \
    cat_mask_test, num_mask_test, num_categories, num_continuous, categorical_columns, continuous_columns = prepare_tabular_dataset(device=torch.device("cpu"))

# Create TensorDatasets
train_dataset = TensorDataset(X_train_cat, X_train_cont, y_train, cat_mask_train, num_mask_train)
test_dataset = TensorDataset(X_test_cat, X_test_cont, y_test, cat_mask_test, num_mask_test)

# DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=32, num_workers=4, pin_memory=True, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, num_workers=4, pin_memory=True, shuffle=False)


model = FTTransformer(
    categories=num_categories,
    num_continuous=len(continuous_columns),
    dim=64,
    dim_out=5,  # Number of unique classes
    depth=8,
    heads=12,
    attn_dropout = 0.1,                 # post-attention dropout
    ff_dropout = 0.1                    # feed forward dropout
).to(device)


def multi_class_accuracy(y_pred, y_true):
    # Get the class predictions from the maximum value of the logits
    _, predicted = torch.max(y_pred, 1)
    correct = (predicted == y_true).float()  # Compare with true labels
    acc = correct.sum() / len(correct)  # Calculate accuracy
    return acc

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

num_epochs = 30

for _, epoch in enumerate(tqdm(range(num_epochs))):
    model.train()
    total_loss = 0
    total_acc = 0
    count = 0

    # Training loop
    train_iterator = tqdm(train_dataloader, desc=f'Epoch {epoch}/{num_epochs} - Train')
    for cat_features, cont_features, labels, masks, _ in train_iterator:
        cat_features, cont_features, labels, masks = cat_features.to(device), cont_features.to(device), labels.to(device), masks.to(device)

        optimizer.zero_grad()
        # Forward pass with masks
        outputs = model(cat_features, cont_features)
        loss = criterion(outputs, labels.squeeze())
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        
        acc = multi_class_accuracy(outputs, labels.squeeze())
        total_loss += loss.item()
        total_acc += acc.item()
        count += 1
        
        # Update tqdm description with current loss and accuracy
        train_iterator.set_description(f'Epoch {epoch}/{num_epochs} - Train Loss: {total_loss/count:.4f}, Acc: {total_acc/count:.4f}')
        
    
    average_loss = total_loss / count
    average_acc = total_acc / count
    # print(f'End of Epoch {epoch}, Train Loss: {average_loss:.4f}, Train Accuracy: {average_acc:.4f}')
    
    # Validation loop every 5 epochs
    if epoch % 2 == 0:
        model.eval()
        total_val_loss = 0
        total_val_acc = 0
        val_count = 0
        
        # Wrap the dataloader with tqdm for the validation loop
        val_iterator = tqdm(test_dataloader, desc=f'Epoch {epoch}/{num_epochs} - Validate')
        with torch.no_grad():
            for val_cat_features, val_cont_features, val_labels, val_masks, _ in val_iterator:
                val_cat_features, val_cont_features, val_labels, val_masks = val_cat_features.to(device), val_cont_features.to(device), val_labels.to(device), val_masks.to(device)
                
                # Forward pass without masks
                val_outputs = model(val_cat_features, val_cont_features)
                val_loss = criterion(val_outputs, val_labels.squeeze())
                
                val_acc = multi_class_accuracy(val_outputs, val_labels.squeeze())
                total_val_loss += val_loss.item()
                total_val_acc += val_acc.item()
                val_count += 1
                
                # Update tqdm description with current validation loss and accuracy
                val_iterator.set_description(f'Epoch {epoch}/{num_epochs} - Val Loss: {total_val_loss/val_count:.4f}, Acc: {total_val_acc/val_count:.4f}')
        
        average_val_loss = total_val_loss / val_count
        average_val_acc = total_val_acc / val_count
        # print(f'End of Epoch {epoch}, Val Loss: {average_val_loss:.4f}, Val Accuracy: {average_val_acc:.4f}')
