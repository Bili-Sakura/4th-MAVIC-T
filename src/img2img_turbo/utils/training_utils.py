"""Training utilities ported from ``vendor/Img2Image-Turbo/src/my_utils/training_utils.py``.

Contains dataset classes for paired/unpaired image-to-image training and
image preprocessing helpers used by the Pix2Pix-Turbo and CycleGAN-Turbo
training scripts.
"""

from __future__ import annotations

import json
import os
import random
from glob import glob

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image


# ---------------------------------------------------------------------------
# Image transforms
# ---------------------------------------------------------------------------

def build_transform(image_prep: str):
    """Build a torchvision transform pipeline for the given *image_prep* string.

    Supported values:
    * ``"resized_crop_512"`` – Resize (shortest edge) then centre-crop to 512.
    * ``"resize_286_randomcrop_256x256_hflip"`` – Standard pix2pix augmentation.
    * ``"resize_256"`` / ``"resize_256x256"``
    * ``"resize_512"`` / ``"resize_512x512"``
    * ``"no_resize"``
    """
    if image_prep == "resized_crop_512":
        T = transforms.Compose([
            transforms.Resize(512, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(512),
        ])
    elif image_prep == "resize_286_randomcrop_256x256_hflip":
        T = transforms.Compose([
            transforms.Resize((286, 286), interpolation=Image.LANCZOS),
            transforms.RandomCrop((256, 256)),
            transforms.RandomHorizontalFlip(),
        ])
    elif image_prep in ["resize_256", "resize_256x256"]:
        T = transforms.Compose([
            transforms.Resize((256, 256), interpolation=Image.LANCZOS)
        ])
    elif image_prep in ["resize_512", "resize_512x512"]:
        T = transforms.Compose([
            transforms.Resize((512, 512), interpolation=Image.LANCZOS)
        ])
    elif image_prep == "no_resize":
        T = transforms.Lambda(lambda x: x)
    else:
        raise ValueError(f"Unknown image_prep: {image_prep}")
    return T


# ---------------------------------------------------------------------------
# Paired dataset (for Pix2Pix-Turbo)
# ---------------------------------------------------------------------------

class PairedDataset(torch.utils.data.Dataset):
    """Dataset for paired image-to-image training (Pix2Pix-Turbo).

    Expects the dataset folder to have the layout::

        dataset_folder/
        ├── train_A/   (source images)
        ├── train_B/   (target images)
        ├── test_A/
        ├── test_B/
        ├── train_prompts.json
        └── test_prompts.json
    """

    def __init__(self, dataset_folder, split, image_prep, tokenizer):
        super().__init__()
        if split == "train":
            self.input_folder = os.path.join(dataset_folder, "train_A")
            self.output_folder = os.path.join(dataset_folder, "train_B")
            captions = os.path.join(dataset_folder, "train_prompts.json")
        elif split == "test":
            self.input_folder = os.path.join(dataset_folder, "test_A")
            self.output_folder = os.path.join(dataset_folder, "test_B")
            captions = os.path.join(dataset_folder, "test_prompts.json")
        with open(captions, "r") as f:
            self.captions = json.load(f)
        self.img_names = list(self.captions.keys())
        self.T = build_transform(image_prep)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        input_img = Image.open(os.path.join(self.input_folder, img_name))
        output_img = Image.open(os.path.join(self.output_folder, img_name))
        caption = self.captions[img_name]

        img_t = self.T(input_img)
        img_t = F.to_tensor(img_t)
        output_t = self.T(output_img)
        output_t = F.to_tensor(output_t)
        output_t = F.normalize(output_t, mean=[0.5], std=[0.5])

        input_ids = self.tokenizer(
            caption, max_length=self.tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids

        return {
            "output_pixel_values": output_t,
            "conditioning_pixel_values": img_t,
            "caption": caption,
            "input_ids": input_ids,
        }


# ---------------------------------------------------------------------------
# Unpaired dataset (for CycleGAN-Turbo)
# ---------------------------------------------------------------------------

class UnpairedDataset(torch.utils.data.Dataset):
    """Dataset for unpaired image-to-image training (CycleGAN-Turbo).

    Expects the dataset folder to have the layout::

        dataset_folder/
        ├── train_A/   (source domain images)
        ├── train_B/   (target domain images)
        ├── test_A/
        ├── test_B/
        ├── fixed_prompt_a.txt
        └── fixed_prompt_b.txt
    """

    def __init__(self, dataset_folder, split, image_prep, tokenizer):
        super().__init__()
        if split == "train":
            self.source_folder = os.path.join(dataset_folder, "train_A")
            self.target_folder = os.path.join(dataset_folder, "train_B")
        elif split == "test":
            self.source_folder = os.path.join(dataset_folder, "test_A")
            self.target_folder = os.path.join(dataset_folder, "test_B")
        self.tokenizer = tokenizer
        with open(os.path.join(dataset_folder, "fixed_prompt_a.txt"), "r") as f:
            self.fixed_caption_src = f.read().strip()
            self.input_ids_src = self.tokenizer(
                self.fixed_caption_src, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids

        with open(os.path.join(dataset_folder, "fixed_prompt_b.txt"), "r") as f:
            self.fixed_caption_tgt = f.read().strip()
            self.input_ids_tgt = self.tokenizer(
                self.fixed_caption_tgt, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids

        self.l_imgs_src = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]:
            self.l_imgs_src.extend(glob(os.path.join(self.source_folder, ext)))
        self.l_imgs_tgt = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]:
            self.l_imgs_tgt.extend(glob(os.path.join(self.target_folder, ext)))
        self.T = build_transform(image_prep)

    def __len__(self):
        return len(self.l_imgs_src) + len(self.l_imgs_tgt)

    def __getitem__(self, index):
        if index < len(self.l_imgs_src):
            img_path_src = self.l_imgs_src[index]
        else:
            img_path_src = random.choice(self.l_imgs_src)
        img_path_tgt = random.choice(self.l_imgs_tgt)
        img_pil_src = Image.open(img_path_src).convert("RGB")
        img_pil_tgt = Image.open(img_path_tgt).convert("RGB")
        img_t_src = F.to_tensor(self.T(img_pil_src))
        img_t_tgt = F.to_tensor(self.T(img_pil_tgt))
        img_t_src = F.normalize(img_t_src, mean=[0.5], std=[0.5])
        img_t_tgt = F.normalize(img_t_tgt, mean=[0.5], std=[0.5])
        return {
            "pixel_values_src": img_t_src,
            "pixel_values_tgt": img_t_tgt,
            "caption_src": self.fixed_caption_src,
            "caption_tgt": self.fixed_caption_tgt,
            "input_ids_src": self.input_ids_src,
            "input_ids_tgt": self.input_ids_tgt,
        }
