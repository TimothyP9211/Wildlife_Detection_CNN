from os import path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import nms
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet50_Weights, resnet50

import numpy as np
import os

from tqdm import tqdm

from detector_model import DetectorModel
from IoU import IoUModule
from dataset import CustomDataset
from PIL import Image
from PIL import ImageDraw
from img_draw import DrawModule

def match_bboxes(IoU, threshold=0.5):
    # Matches predicted boxes to target boxes above a set threshold
    matches = []
    matched_pred = set()
    matched_target = set()

    candidates = []

    for i in range(IoU.shape[0]):
        for j in range(IoU.shape[1]):
            candidates.append((IoU[i, j].item(), i, j))

    # Sort candidates by decreasing IoU to prioritize higher matches
    candidates.sort(reverse=True)

    for iou, pred_idx, target_idx in candidates:
        # Only consider matches above threshold and not already matched
        if iou < threshold:
            continue
        if (pred_idx in matched_pred) or (target_idx in matched_target):
            continue

        matched_pred.add(pred_idx)
        matched_target.add(target_idx)
        matches.append((pred_idx, target_idx, iou))
    
    return matches

def match_bboxes_best(IoU):
    # Matches each target box to the predicted box with the highest IoU without a threshold
    matches = []
    if IoU.numel() == 0:
        return matches

    num_pred_boxes, num_target_boxes = IoU.shape

    used_preds = set()
    for target_idx in range(num_target_boxes):
        best_pred_idx = torch.argmax(IoU[:, target_idx]).item()
        best_iou = IoU[best_pred_idx, target_idx].item()

        if best_pred_idx not in used_preds:
            matches.append((best_pred_idx, target_idx, best_iou))
            used_preds.add(best_pred_idx)

    return matches

# Train over one epoch
def train_model(model, dataLoader, optimizer, device, iou_module,
          reg_criterion, class_criterion, objectness_criterion, train_loss_logger, epoch=None):
    model.train()
    running_loss = 0.0

    # tqdm for visualizing progress
    if epoch is not None:
        loop = tqdm(dataLoader, desc=f"Training Epoch {epoch}", leave=True)
    else:
        loop = tqdm(dataLoader, desc="Training", leave=True)

    for batch_idx, (images, target_bboxes, target_labels) in enumerate(loop):
        images, target_bboxes, target_labels = images.to(device), target_bboxes.to(device), target_labels.to(device).long()
        optimizer.zero_grad()
        pred_bboxes, class_logits, objectness_logits = model(images)

        batch_size = images.shape[0]
        match_count = 0
        object_exists = torch.zeros_like(objectness_logits)
        reg_loss = torch.tensor(0.0, device=device)
        class_loss = torch.tensor(0.0, device=device)

        # Iterate over the batch and compute loss for bounding boxes and labels
        for i in range(batch_size):
            # Ignore padded boxes
            real_target_bboxes = target_bboxes[i][target_labels[i] != -1]
            real_target_labels = target_labels[i][target_labels[i] != -1]

            IoU = iou_module(pred_bboxes[i], real_target_bboxes)
            # Match bounding boxes without thresholding since early predictions may suck
            matches = match_bboxes_best(IoU)

            for pred_idx, target_idx, iou_score in matches:
                # Object found in this box
                object_exists[i, pred_idx] = 1

                # Compute regression and classification loss for this matched box
                reg_loss += reg_criterion(pred_bboxes[i, pred_idx], real_target_bboxes[target_idx])
                class_loss += class_criterion(class_logits[i, pred_idx].unsqueeze(0), real_target_labels[target_idx].unsqueeze(0))
                match_count += 1

        # If no matches, then train only for objectness loss
        if match_count > 0:
            reg_loss /= match_count
            class_loss /= match_count
        else:
            reg_loss = torch.tensor(0.0, device=device)
            class_loss = torch.tensor(0.0, device=device)
        
        objectness_loss = objectness_criterion(objectness_logits, object_exists)

        loss = 4 * reg_loss + class_loss + objectness_loss

        # Backpropagation and optimization step
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        # Log average per batch train loss for plotting
        train_loss_logger.append(running_loss / (batch_idx + 1))
    
    return running_loss / len(dataLoader)

# Validate over one epoch
def evaluate_model(model, dataLoader, device, iou_module, threshold,
                reg_criterion, class_criterion, objectness_criterion, val_loss_logger, epoch=None):
    model.eval()
    running_loss = 0.0

    # tqdm for visualizing progress
    if epoch is not None:
        loop = tqdm(dataLoader, desc=f"Validation Epoch {epoch}", leave=True)
    else:
        loop = tqdm(dataLoader, desc="Validation", leave=True)

    with torch.no_grad():
        for batch_idx, (images, target_bboxes, target_labels) in enumerate(loop):
            images, target_bboxes, target_labels = images.to(device), target_bboxes.to(device), target_labels.to(device).long()
            pred_bboxes, class_logits, objectness_logits = model(images)

            batch_size = images.shape[0]
            match_count = 0
            object_exists = torch.zeros_like(objectness_logits)
            reg_loss = torch.tensor(0.0, device=device)
            class_loss = torch.tensor(0.0, device=device)

            # Iterate over the batch and compute loss for bounding boxes and labels
            for i in range(batch_size):
                # Ignore padded boxes
                real_target_bboxes = target_bboxes[i][target_labels[i] != -1]
                real_target_labels = target_labels[i][target_labels[i] != -1]

                IoU = iou_module(pred_bboxes[i], real_target_bboxes)
                # Use thresholded matching for evaluation to reflect actual performance at a given IoU threshold
                matches = match_bboxes(IoU, threshold=threshold)

                for pred_idx, target_idx, iou_score in matches:
                    # Object found in this box
                    object_exists[i, pred_idx] = 1

                    # Compute regression and classification loss for this matched box
                    reg_loss += reg_criterion(pred_bboxes[i, pred_idx], real_target_bboxes[target_idx])
                    class_loss += class_criterion(class_logits[i, pred_idx].unsqueeze(0), real_target_labels[target_idx].unsqueeze(0))
                    match_count += 1

            # If no matches, then train only for objectness loss
            if match_count > 0:
                reg_loss /= match_count
                class_loss /= match_count
            else:
                reg_loss = torch.tensor(0.0, device=device)
                class_loss = torch.tensor(0.0, device=device)
            
            objectness_loss = objectness_criterion(objectness_logits, object_exists)

            loss = 4 * reg_loss + class_loss + objectness_loss
            running_loss += loss.item()

            # Log average per batch val loss for plotting
            val_loss_logger.append(running_loss / (batch_idx + 1))

    return running_loss / len(dataLoader)

# Custom collate function to pad batches since images have varying numbers of bboxes
def multibox_collate(batch):
    images, boxes, labels = zip(*batch)
    images = torch.stack(images, dim=0)
    batch_size = len(boxes)
    max_num_bboxes = max(b.shape[0] for b in boxes)

    # Image has no boxes
    if max_num_bboxes == 0:
        max_num_bboxes = 1

    # Pad labels with -1 for ignored boxes
    padded_bboxes = torch.zeros((batch_size, max_num_bboxes, 4), dtype=torch.float32)
    padded_labels = torch.full((batch_size, max_num_bboxes), -1, dtype=torch.long)

    for i in range(batch_size):
        num_bboxes = boxes[i].shape[0]
        if num_bboxes > 0:
            padded_bboxes[i, :num_bboxes] = boxes[i]
            padded_labels[i, :num_bboxes] = labels[i]

    return images, padded_bboxes, padded_labels

# Draw the bounding boxes for a single image from the test set
def output_test_image(test_path, model, device, transform, drawmodule, idx, threshold=0.5):
    test_image = Image.open(test_path).convert("RGB")

    with torch.no_grad():
        input_tensor = transform(test_image).unsqueeze(0).to(device)
        pred_bboxes, class_logits, objectness_logits = model(input_tensor)  

    pred_bboxes = pred_bboxes[0]                 
    class_logits = class_logits[0]               
    objectness_logits = objectness_logits[0]

    objectness_scores = torch.sigmoid(objectness_logits).squeeze(-1)
    class_probs = torch.softmax(class_logits, dim=-1)
    class_scores, class_labels = torch.max(class_probs, dim=-1)
    final_scores = objectness_scores * class_scores
    kept_indices = final_scores > threshold

    final_bboxes = pred_bboxes[kept_indices]
    final_labels = class_labels[kept_indices]
    final_scores = final_scores[kept_indices]

    drawmodule.draw_predictions(test_path, final_bboxes, final_labels, final_scores, output_path=f"output/test_output_{idx}.jpg")


def main():
    # Dataset specific parameters
    num_classes = 11
    num_bboxes = 8
    threshold = 0.4

    # Base model for feature extraction
    resnet = resnet50(weights=ResNet50_Weights.DEFAULT)

    # Initialize detector model and use cuda for faster training if available
    model = DetectorModel(baseModel=resnet, numClasses=num_classes, numBBoxes=num_bboxes)
    print(f"cuda available: {torch.cuda.is_available()}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Adam optimizer with dynamic lr
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    iou_module = IoUModule().to(device)
    train_loss_logger, val_loss_logger = [], []

    # Loss functions for each part of detection pipeline
    reg_criterion = nn.SmoothL1Loss()
    class_criterion = nn.CrossEntropyLoss()
    objectness_criterion = nn.BCEWithLogitsLoss()

    # Dataset initialization 
    dataset_root = "detect_dataset"
    train_dataset = CustomDataset(
        images_dir=f"{dataset_root}/images/train",
        labels_dir=f"{dataset_root}/labels/train",
        image_size=224
    )
    val_dataset = CustomDataset(
        images_dir=f"{dataset_root}/images/val",
        labels_dir=f"{dataset_root}/labels/val",
        image_size=224
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=multibox_collate)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=multibox_collate)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = train_model(model, train_loader, optimizer, device, iou_module,
                                 reg_criterion, class_criterion, objectness_criterion, train_loss_logger, epoch=epoch+1)
        val_loss = evaluate_model(model, val_loader, device, iou_module, threshold,
                                  reg_criterion, class_criterion, objectness_criterion, val_loss_logger, epoch=epoch+1)
        scheduler.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    # Run the model on the test images and draw the outputs
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    drawmodule = DrawModule()
    test_names = []
    test_folder = f"{dataset_root}/images/test"
    for filename in os.listdir(test_folder):
        if filename.lower().endswith((".jpg",".png")):
            test_names.append(filename)

    for name in test_names:
        test_path = os.path.join(test_folder, name)
        output_test_image(test_path, model, device, transform, drawmodule, idx=name.split(".")[0], threshold=threshold)

    # plt.plot(train_loss_logger, label = "training losses")
    # plt.plot(val_loss_logger, label = "validation losses")
    # plt.legend()
    # plt.title("losses over epoch")
    # plt.show()

    return

if __name__ == "__main__":
    main()