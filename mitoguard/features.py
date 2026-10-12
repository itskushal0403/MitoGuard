"""Frozen feature extractors for the Stage 2 imposter filter.

All backbones take 224x224 RGB crops (level-0 pixels) centred on a candidate and return one vector per crop.
A feature *spec* is  <model>[_tta][-z<N>][-ctr], e.g. "phikon-z112-ctr":
  - phikon      : owkin/phikon      ViT-B/16, iBOT on 40M TCGA patches (CLS token, 768-d)
  - phikon2     : owkin/phikon-v2   ViT-L/16, DINOv2 on 460M pathology patches (CLS, 1024-d)
  - resnet50    : ImageNet-supervised ResNet-50 (global average pool, 2048-d) - ablation baseline
  - _tta        : mean embedding over the 8 dihedral (flip/rotate) views of the crop
  - -z<N>       : zoom: use only the central N x N pixels, resized to 224 (cell fills more of the view)
  - -ctr        : ViTs only: concatenate the CLS token with the mean of the central 4x4 patch tokens,
                  which cover the candidate cell itself rather than the surrounding tissue
  - uni         : MahmoodLab/UNI    ViT-L/16 (gated; used only if HF access is available)
"""
from __future__ import annotations

import re

import numpy as np
import torch
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Backbone(torch.nn.Module):
    def __init__(self, spec: str, device: str = "cuda"):
        super().__init__()
        m = re.fullmatch(r"([a-z0-9]+)(_tta)?(?:-z(\d+))?(-ctr)?", spec)
        if m is None:
            raise ValueError(f"bad feature spec {spec!r}")
        name, self.tta, zoom, self.ctr = m.group(1), bool(m.group(2)), m.group(3), bool(m.group(4))
        self.zoom = int(zoom) if zoom else 224
        self.spec, self.name = spec, name
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
        if self.ctr:
            if name not in ("phikon", "phikon2"):
                raise ValueError("-ctr needs a ViT backbone")
            self.dim *= 2
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = self.model.to(self.device, self.dtype).eval()
        self.register_buffer("mean", torch.tensor(mean, device=self.device).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, device=self.device).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, crops: np.ndarray) -> np.ndarray:
        """crops: (B, 224, 224, 3) uint8 RGB -> (B, dim) float32."""
        x = torch.from_numpy(crops).to(self.device).permute(0, 3, 1, 2).float() / 255.0
        if self.zoom != x.shape[-1]:
            o = (x.shape[-1] - self.zoom) // 2
            x = F.interpolate(x[..., o:o + self.zoom, o:o + self.zoom], size=224, mode="bilinear", align_corners=False)
        x = ((x - self.mean) / self.std).to(self.dtype)
        if not self.tta:
            return self._embed(x).float().cpu().numpy()
        views = [torch.rot90(v, k, dims=(2, 3)) for v in (x, x.flip(3)) for k in range(4)]
        return torch.stack([self._embed(v).float() for v in views]).mean(0).cpu().numpy()

    def _embed(self, x):
        if self.name in ("phikon", "phikon2"):
            h = self.model(pixel_values=x).last_hidden_state
            if not self.ctr:
                return h[:, 0]
            g = 224 // 16  # 14 x 14 patch grid; patch tokens are the last g*g tokens
            grid = h[:, -g * g:].reshape(len(h), g, g, -1)
            return torch.cat([h[:, 0], grid[:, 5:9, 5:9].mean((1, 2))], 1)
        return self.model(x)
