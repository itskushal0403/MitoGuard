"""Frozen feature extractors for the Stage 2 imposter filter.

All backbones take 224x224 RGB crops centred on a candidate and return one vector per crop.
  - phikon      : owkin/phikon      ViT-B/16, iBOT on 40M TCGA patches (CLS token, 768-d)
  - phikon2     : owkin/phikon-v2   ViT-L/16, DINOv2 on 460M pathology patches (CLS, 1024-d)
  - resnet50    : ImageNet-supervised ResNet-50 (global average pool, 2048-d) - ablation baseline
  - <name>_tta  : any of the above, averaged over the 8 dihedral (flip/rotate) views of the crop
  - uni         : MahmoodLab/UNI    ViT-L/16 (gated; used only if HF access is available)
"""
from __future__ import annotations

import numpy as np
import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Backbone(torch.nn.Module):
    def __init__(self, name: str, device: str = "cuda"):
        super().__init__()
        self.tta = name.endswith("_tta")  # e.g. "phikon_tta": mean embedding over the 8 dihedral transforms
        name = name[:-4] if self.tta else name
        self.name = name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        mean, std = IMAGENET_MEAN, IMAGENET_STD
        if name == "phikon":
            from transformers import ViTModel

            self.model = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
            self.dim = 768
        elif name == "phikon2":
            from transformers import AutoModel

            self.model = AutoModel.from_pretrained("owkin/phikon-v2")
            self.dim = 1024
        elif name == "resnet50":
            import timm

            self.model = timm.create_model("resnet50.tv2_in1k", pretrained=True, num_classes=0)
            self.dim = 2048
        elif name == "uni":
            import timm

            self.model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)
            self.dim = 1024
        else:
            raise ValueError(name)
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = self.model.to(self.device, self.dtype).eval()
        self.register_buffer("mean", torch.tensor(mean, device=self.device).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, device=self.device).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, crops: np.ndarray) -> np.ndarray:
        """crops: (B, 224, 224, 3) uint8 RGB -> (B, dim) float32."""
        x = torch.from_numpy(crops).to(self.device).permute(0, 3, 1, 2).float() / 255.0
        x = ((x - self.mean) / self.std).to(self.dtype)
        if not self.tta:
            return self._embed(x).float().cpu().numpy()
        views = [torch.rot90(v, k, dims=(2, 3)) for v in (x, x.flip(3)) for k in range(4)]
        return torch.stack([self._embed(v).float() for v in views]).mean(0).cpu().numpy()

    def _embed(self, x):
        if self.name in ("phikon", "phikon2"):
            return self.model(pixel_values=x).last_hidden_state[:, 0]
        return self.model(x)
