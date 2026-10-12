"""Stage 2 v2: parameter-efficient fine-tuning of a pathology foundation model as a cell classifier.

* Backbones: Phikon (ViT-B/16), Phikon-v2 (ViT-L/16), Kaiko Midnight (ViT-g/14), Lunit DINO (ViT-S/8),
  Kaiko ViT-B/16. LoRA adapters on the attention projections + a trainable head; the backbone stays frozen.
* Cell-centred read-out: [CLS || mean of the central patch tokens], for any patch grid (14x14 .. 28x28).
* Domain randomisation on the GPU: per-sample H&E optical-density jitter, brightness/contrast/saturation,
  blur (focus), dihedral transforms, translation (+-12 px, detector localisation error) and scale (+-15%)
  jitter, all applied to cached 224 px level-0 crops; the zoomed 112 px centre is resampled to 224 px.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

from .stain import _HED_FROM_RGB, _RGB_FROM_HED

SPECS = {
    # name: (loader, hf/timm id, patch, lora target modules, mean, std)
    "phikon": ("hf_vit", "owkin/phikon", 16, ["query", "value"], (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "phikon2": ("hf_auto", "owkin/phikon-v2", 16, ["query", "value"], (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "midnight": ("hf_auto", "kaiko-ai/midnight", 14, ["query", "value"], (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    "lunit": ("timm", "hf-hub:1aurent/vit_small_patch8_224.lunit_dino", 8, ["qkv"],
              (0.70322989, 0.53606487, 0.66096631), (0.21716536, 0.26081574, 0.20723464)),
    "kaikob": ("timm", "hf-hub:1aurent/vit_base_patch16_224.kaiko_ai_towards_large_pathology_fms", 16, ["qkv"],
               (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
}


def load_backbone(name):
    kind, ident, patch, targets, mean, std = SPECS[name]
    if kind == "hf_vit":
        from transformers import ViTModel
        model, dim = ViTModel.from_pretrained(ident, add_pooling_layer=False), None
    elif kind == "hf_auto":
        from transformers import AutoModel
        model, dim = AutoModel.from_pretrained(ident), None
    else:
        import timm
        model = timm.create_model(ident, pretrained=True, num_classes=0, dynamic_img_size=True)
        dim = model.num_features
    if dim is None:
        dim = model.config.hidden_size
    return kind, model, dim, patch, targets, mean, std


class CellClassifier(nn.Module):
    def __init__(self, name: str, lora_r: int = 16, zoom: int = 112, center_frac: float = 0.3):
        super().__init__()
        from peft import LoraConfig, get_peft_model

        self.name, self.zoom = name, zoom
        self.kind, model, dim, self.patch, targets, mean, std = load_backbone(name)
        for p in model.parameters():
            p.requires_grad = False
        cfg = LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.1, target_modules=targets, bias="none")
        self.backbone = get_peft_model(model, cfg)
        g = 224 // self.patch
        a = int(round(g * (0.5 - center_frac / 2)))
        self.grid, self.c0, self.c1 = g, a, g - a
        self.head = nn.Sequential(nn.LayerNorm(2 * dim), nn.Dropout(0.2), nn.Linear(2 * dim, 1))
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def tokens(self, x):
        if self.kind.startswith("hf"):
            h = self.backbone(pixel_values=x).last_hidden_state
        else:
            h = self.backbone.base_model.model.forward_features(x)
        return h

    def forward(self, x):
        """x: (B, 3, 224, 224) float in [0, 1] (already zoomed / augmented)."""
        x = (x - self.mean) / self.std
        h = self.tokens(x)
        g = self.grid
        grid = h[:, -g * g:].reshape(len(h), g, g, -1)
        ctr = grid[:, self.c0:self.c1, self.c0:self.c1].mean((1, 2))
        return self.head(torch.cat([h[:, 0], ctr], 1).float()).squeeze(-1)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


# ----------------------------------------------------------------------------------------------- augmentation
_HED = torch.tensor(_HED_FROM_RGB)
_RGB = torch.tensor(_RGB_FROM_HED)


def zoom_crop(x, zoom, train, max_shift=12.0, scale=0.15):
    """x: (B, 3, H, W) float. Resample the (jittered) central zoom x zoom window to 224 x 224."""
    b, _, h, w = x.shape
    cx, cy = torch.full((b,), w / 2, device=x.device), torch.full((b,), h / 2, device=x.device)
    size = torch.full((b,), float(zoom), device=x.device)
    if train:
        cx = cx + (torch.rand(b, device=x.device) * 2 - 1) * max_shift
        cy = cy + (torch.rand(b, device=x.device) * 2 - 1) * max_shift
        size = size * (1 + (torch.rand(b, device=x.device) * 2 - 1) * scale)
    boxes = torch.stack([torch.arange(b, device=x.device, dtype=x.dtype), cx - size / 2, cy - size / 2,
                         cx + size / 2, cy + size / 2], 1)
    return roi_align(x, boxes, output_size=224, spatial_scale=1.0, sampling_ratio=2, aligned=True)


def stain_jitter(x, sigma=0.08, bias=0.08):
    """Per-sample H&E optical-density jitter on a (B, 3, H, W) [0, 1] batch."""
    b = x.shape[0]
    dev = x.device
    hed_m, rgb_m = _HED.to(dev), _RGB.to(dev)
    alpha = 1 + (torch.rand(b, 1, 3, device=dev) * 2 - 1) * sigma
    beta = (torch.rand(b, 1, 3, device=dev) * 2 - 1) * bias
    od = -torch.log((x * 255 + 1) / 256).permute(0, 2, 3, 1).reshape(b, -1, 3)
    od2 = ((od @ hed_m) * alpha + beta) @ rgb_m
    out = (256 * torch.exp(-od2) - 1) / 255
    return out.reshape(b, x.shape[2], x.shape[3], 3).permute(0, 3, 1, 2).clamp(0, 1)


def photometric(x):
    b = x.shape[0]
    dev = x.device
    bright = 1 + (torch.rand(b, 1, 1, 1, device=dev) * 2 - 1) * 0.15
    contrast = 1 + (torch.rand(b, 1, 1, 1, device=dev) * 2 - 1) * 0.15
    sat = 1 + (torch.rand(b, 1, 1, 1, device=dev) * 2 - 1) * 0.2
    mean = x.mean((1, 2, 3), keepdim=True)
    x = (x - mean) * contrast + mean
    gray = x.mean(1, keepdim=True)
    x = (x - gray) * sat + gray
    return (x * bright).clamp(0, 1)


def random_blur(x, p=0.25):
    m = torch.rand(x.shape[0], device=x.device) < p
    if m.any():
        sigma = float(np.random.uniform(0.5, 1.5))
        k = 2 * math.ceil(2 * sigma) + 1
        import torchvision.transforms.functional as TF
        x = x.clone()
        x[m] = TF.gaussian_blur(x[m], [k, k], [sigma, sigma])
    return x


def dihedral(x, k):
    """k in 0..7: rotate by 90*(k%4), flip if k>=4."""
    if k >= 4:
        x = x.flip(3)
    return torch.rot90(x, k % 4, dims=(2, 3))


def random_dihedral(x):
    k = torch.randint(0, 8, (x.shape[0],), device=x.device)
    out = torch.empty_like(x)
    for v in range(8):
        m = k == v
        if m.any():
            out[m] = dihedral(x[m], v)
    return out


def augment(u8, zoom):
    """u8: (B, 224, 224, 3) uint8 on device -> augmented (B, 3, 224, 224) float."""
    x = u8.permute(0, 3, 1, 2).float() / 255
    x = stain_jitter(x)
    x = zoom_crop(x, zoom, train=True)
    x = photometric(x)
    x = random_blur(x)
    return random_dihedral(x)


@torch.no_grad()
def predict(model, u8_batches, zoom, tta=8, device="cuda"):
    """Mean sigmoid probability over `tta` dihedral views. u8_batches yields uint8 (B,224,224,3) tensors."""
    model.eval()
    out = []
    for u8 in u8_batches:
        x = zoom_crop(u8.to(device).permute(0, 3, 1, 2).float() / 255, zoom, train=False)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = torch.stack([model(dihedral(x, k)) for k in range(tta)])
        out.append(torch.sigmoid(logits.float()).mean(0).cpu())
    return torch.cat(out).numpy() if out else np.zeros(0, np.float32)
