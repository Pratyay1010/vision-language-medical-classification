import os
import numpy as np
import pandas as pd
from PIL import Image

from torch.utils.data import Dataset, Subset
from torchvision import transforms


# -----------------------------
# Transforms
# -----------------------------

def get_imagenet_transforms(resize_size=224):
    train_tfms = transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_tfms = transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_tfms, test_tfms


def get_orthonet_augmented_transforms(resize_size=224):
    train_tfms = transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_tfms = transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_tfms, test_tfms


def get_pacemaker_transforms(resize_size=224):
    train_tfms = transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    test_tfms = transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    return train_tfms, test_tfms


# -----------------------------
# OrthoNet Dataset
# -----------------------------

class OrthonetDataset(Dataset):
    """OrthoNet dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        self.label2idx = {
            label: idx
            for idx, label in enumerate(self.data["labels"].unique())
        }

        self.data["label_idx"] = self.data["labels"].map(self.label2idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        img_path = os.path.join(
            self.root_dir,
            row["filenames"]
        )

        image = Image.open(img_path).convert("RGB")
        label = row["label_idx"]

        if self.transform:
            image = self.transform(image)

        return image, label

    @property
    def classes(self):
        inv_map = {v: k for k, v in self.label2idx.items()}
        return [inv_map[i] for i in sorted(inv_map.keys())]


# -----------------------------
# Pacemaker Dataset
# -----------------------------

class PacemakerDataset(Dataset):
    """Pacemaker dataset."""

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.classes = sorted(os.listdir(root_dir))

        self.label2idx = {
            cls_name: idx
            for idx, cls_name in enumerate(self.classes)
        }

        self.samples = []

        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)

            for fname in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, fname)

                self.samples.append(
                    (img_path, self.label2idx[cls_name])
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# -----------------------------
# Manufacturer Mapping
# -----------------------------

def get_manufacturer(class_name):
    return class_name.split("-")[0].strip().upper()


class MappedImageFolder(Dataset):
    """Maps pacemaker subclasses to manufacturers."""

    def __init__(self, image_folder_dataset, manufacturer_list):
        self.dataset = image_folder_dataset
        self.manufacturer_list = manufacturer_list
        self.class_to_manufacturer = self._build_mapping()

    def _build_mapping(self):
        mapping = {}

        for idx, class_name in enumerate(self.dataset.classes):
            manufacturer = get_manufacturer(class_name)

            mapping[idx] = (
                self.manufacturer_list.index(manufacturer)
                if manufacturer in self.manufacturer_list
                else 0
            )

        return mapping

    def __getitem__(self, index):
        img, original_label = self.dataset[index]
        mapped_label = self.class_to_manufacturer[original_label]

        return img, mapped_label

    def __len__(self):
        return len(self.dataset)


# -----------------------------
# Dataset Split Utility
# -----------------------------

def split_dataset_by_labels(
    dataset,
    test_size=0.2,
    random_state=42
):
    from sklearn.model_selection import train_test_split

    targets = [label for _, label in dataset]
    indices = np.arange(len(dataset))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=targets,
        random_state=random_state
    )

    return (
        Subset(dataset, train_idx),
        Subset(dataset, val_idx)
    )