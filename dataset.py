import os
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CustomDataset(Dataset):
    # Dataset in YOLO format: images and labels in seperate folders split into train/val/test with labels in .txt format

    def __init__(self, images_dir, labels_dir, image_size=224):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.image_paths = sorted([
            p for p in self.images_dir.iterdir()
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ])

        # Transformation to resize images (default 224x224 used by resnet50)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        label_path = self.labels_dir / f"{image_path.stem}.txt"
        boxes = []
        labels = []

        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()

                    if len(parts) != 5:
                        continue

                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                    # Convert from (x_center, y_center, width, height) to (x_min, y_min, width, height)
                    x_min = x_center - width / 2
                    y_min = y_center - height / 2

                    boxes.append([x_min, y_min, width, height])
                    labels.append(class_id)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        return image, boxes, labels