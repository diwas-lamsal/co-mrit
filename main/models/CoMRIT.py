import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from .clip_loss import ClipLoss

class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim,
        dropout = 0.1,
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class CoMRIT(nn.Module):
    def __init__(self, image_encoder, tabular_encoder, image_feature_dim, tabular_feature_dim, projection_dim, 
                 logit_scale = np.log(1 / 0.07), parallel_classify=False, num_classes=4):
        super().__init__()
        self.image_encoder = image_encoder
        self.tabular_encoder = tabular_encoder
        self.image_projection = ProjectionHead(image_feature_dim, projection_dim)
        self.tabular_projection = ProjectionHead(tabular_feature_dim, projection_dim)
        self.logit_scale = torch.tensor(logit_scale)
        self.loss = ClipLoss()
        
        self.parallel_classify = parallel_classify
        if parallel_classify:
            self.classification_head = nn.Sequential(
                nn.LayerNorm(image_feature_dim),
                nn.Linear(image_feature_dim, num_classes)
            )
    
    def forward(self, images, cat_features, cont_features, labels=None):
        image_features = self.image_encoder(images)
        tabular_features = self.tabular_encoder(cat_features, cont_features)
        
        # Project both types of features into the same space
        image_embeddings = self.image_projection(image_features)
        tabular_embeddings = self.tabular_projection(tabular_features)
        
        # L2 norm for cosine similarity
        image_embeddings = F.normalize(image_embeddings, p=2)
        tabular_embeddings = F.normalize(tabular_embeddings,p=2)
        
        contrastive_loss = self.loss(image_features=image_embeddings, text_features=tabular_embeddings, logit_scale=self.logit_scale)
        output = {'contrastive_loss': contrastive_loss}

        if self.parallel_classify:
            classification_logits = self.classification_head(image_features)
            classification_loss = F.cross_entropy(classification_logits, labels)
            output['classification_loss'] = classification_loss
            output['classification_logits'] = classification_logits
        
        return output
