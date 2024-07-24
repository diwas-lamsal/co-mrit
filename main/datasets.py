from torch.utils.data import Dataset
import torch

class MRIDataset(Dataset):
    def __init__(self, image_paths, labels, transforms, path_prefix = '/home/media/dataset/mri-ssd/') -> None:
        super().__init__()
        self.image_paths = image_paths
        self.labels = labels
        self.transforms = transforms
        self.path_prefix = path_prefix
        
    def __getitem__(self, index):
        image_data = {'path': self.path_prefix+self.image_paths.iloc[index]['path'][29:]} # Modified to make it work with the new paths from csv
        image = self.transforms(image_data)['path']
        label = self.labels[index]
        return image, label
    
    def __len__(self):
        return len(self.labels)


class CombinedDataset(Dataset):
    def __init__(self, image_paths, cat_features, cont_features, labels, cat_masks, num_masks, transforms, path_prefix = '/home/media/dataset/mri-ssd/'):
        super().__init__()
        self.image_paths = image_paths
        self.cat_features = cat_features
        self.cont_features = cont_features
        self.labels = labels
        self.cat_masks = cat_masks
        self.num_masks = num_masks
        self.transforms = transforms
        self.path_prefix = path_prefix
        
    def __getitem__(self, index):
        # Apply MONAI transforms to load and process the image
        image_data = {'path': self.path_prefix+self.image_paths.iloc[index]['path'][29:]} # Modified to make it work with the new paths from csv
        image = self.transforms(image_data)['path']
        
        # Temporarily, set random tensor
        # image = torch.rand((1,96,96,96), dtype=torch.float32)
        
        # Retrieve tabular data
        cat = self.cat_features[index]
        cont = self.cont_features[index]
        label = self.labels[index]
        cat_mask = self.cat_masks[index]
        num_mask = self.num_masks[index]
        
        return image, cat, cont, label, cat_mask, num_mask

    def __len__(self):
        return len(self.labels)
