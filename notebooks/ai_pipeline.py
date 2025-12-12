import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

# ResNet-18 based feature extractor
class ResNet18Embedder:
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # (B, 512, 1, 1)
        self.backbone.to(self.device)
        self.backbone.eval()

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225],
            ),
        ])

    # Encode an image to feature vector
    @torch.no_grad()
    def encode_pil(self, img): 
        x = self.transform(img).unsqueeze(0).to(self.device)  # (1, 3, 224, 224)
        feat = self.backbone(x)  # (1, 512, 1, 1)
        feat = feat.view(-1).float()  # (512,)
        feat = feat / (feat.norm(p=2) + 1e-8)             
        return feat.detach().cpu().numpy().astype(np.float32)

    # Extract ResNet features for a list of items
    def extract_features(self, items, viz_step=200):
        feats = []
        # Iterate over items and compute features
        for i, it in enumerate(items):
            if viz_step and i % viz_step == 0:
                print(f"Processing {i} / {len(items)}")
            img = Image.open(it["path"]).convert("RGB")
            feats.append(self.encode_pil(img))
        return np.stack(feats, axis=0)