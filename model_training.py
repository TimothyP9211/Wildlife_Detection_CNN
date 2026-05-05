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

from matplotlib import pyplot as plt

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

        loss = 2 * reg_loss + class_loss + objectness_loss

        # Backpropagation and optimization step
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        # Log average per batch train loss for plotting
        train_loss_logger.append(running_loss / (batch_idx + 1))
    
    return running_loss / len(dataLoader)

# Validate over one epoch
def evaluate_model(model, dataLoader, device, iou_module, iou_threshold, confidence_threshold, epoch=None,
                   val_precision_logger=None, val_recall_logger=None, val_miou_logger=None, val_class_accuracy_logger=None):
    model.eval()

    # Metrics
    total_matches = 0
    total_predictions = 0
    total_targets = 0
    total_iou = 0.0
    correct_classes = 0

    # tqdm for visualizing progress
    if epoch is not None:
        loop = tqdm(dataLoader, desc=f"Validation Epoch {epoch}", leave=True)
    else:
        loop = tqdm(dataLoader, desc="Validation", leave=True)

    with torch.no_grad():
        for images, target_bboxes, target_labels in loop:
            images, target_bboxes, target_labels  = images.to(device), target_bboxes.to(device), target_labels.to(device).long()
            pred_bboxes, class_logits, objectness_logits = model(images)

            objectness_scores = torch.sigmoid(objectness_logits).squeeze(-1)
            class_probs = torch.softmax(class_logits, dim=-1)
            class_scores, pred_labels = torch.max(class_probs, dim=-1)
            final_scores = objectness_scores * class_scores

            batch_size = images.shape[0]

            for i in range(batch_size):
                valid_target_mask = target_labels[i] != -1
                real_target_bboxes = target_bboxes[i][valid_target_mask]
                real_target_labels = target_labels[i][valid_target_mask]

                # Filter our images below the confidence threshold before evaluation
                keep = final_scores[i] > confidence_threshold
                kept_pred_bboxes = pred_bboxes[i][keep]
                kept_pred_labels = pred_labels[i][keep]

                num_targets = real_target_bboxes.shape[0]
                num_predictions = kept_pred_bboxes.shape[0]
                total_targets += num_targets
                total_predictions += num_predictions

                # Skip batch if no targets
                if num_targets == 0 or num_predictions == 0:
                    continue

                IoU = iou_module(kept_pred_bboxes, real_target_bboxes)

                matches = match_bboxes(IoU, threshold=iou_threshold)

                total_matches += len(matches)

                # Count the number of correct class predictions and sum IoU scores for matched boxes
                for pred_idx, target_idx, iou_score in matches:
                    total_iou += iou_score
                    if kept_pred_labels[pred_idx] == real_target_labels[target_idx]:
                        correct_classes += 1

            # Evaluation metrics 

            # % of correct predictions
            precision = total_matches / (total_predictions + 1e-5)

            # % of targets detected
            recall = total_matches / (total_targets + 1e-5)

            # Average IoU across matched boxes
            mean_iou = total_iou / (total_matches + 1e-5)

            # % of matched boxes with correct class prediction
            class_accuracy = correct_classes / (total_matches + 1e-5)

            loop.set_postfix({
                "precision": f"{precision:.3f}",
                "recall": f"{recall:.3f}",
                "mIoU": f"{mean_iou:.3f}",
                "cls_acc": f"{class_accuracy:.3f}"
            })

            # Log validation metrics for plotting
            if val_precision_logger is not None:
                val_precision_logger.append(precision)
            if val_recall_logger is not None:
                val_recall_logger.append(recall)
            if val_miou_logger is not None:
                val_miou_logger.append(mean_iou)
            if val_class_accuracy_logger is not None:
                val_class_accuracy_logger.append(class_accuracy)

    precision = total_matches / (total_predictions + 1e-5)
    recall = total_matches / (total_targets + 1e-5)
    mean_iou = total_iou / (total_matches + 1e-5)
    class_accuracy = correct_classes / (total_matches + 1e-5)

    return {"precision": precision, "recall": recall,"mean_iou": mean_iou,"class_accuracy": class_accuracy,}

# Custom collate function to pad batches since images may have varying numbers of bboxes
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

# Draw the bounding boxes for a single image from the test set and save the output
def output_test_image(test_path, model, device, transform, drawmodule, idx, threshold):
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
    num_classes = 7
    num_bboxes = 8
    threshold = 0.45

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
    train_loss_logger, val_precision_logger, val_recall_logger, val_miou_logger, val_class_accuracy_logger = [], [], [], [], []

    # Loss functions for each part of detection pipeline
    reg_criterion = nn.SmoothL1Loss()
    class_criterion = nn.CrossEntropyLoss()
    objectness_criterion = nn.BCEWithLogitsLoss()

    # Dataset initialization 

    # Augmentation for training set, does not alter bounding boxes (ie no geometric transforms)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.95)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])

    # Dataset structure in typical format for YOLO datasets
    dataset_root = "detect_dataset"
    train_dataset = CustomDataset(
        images_dir=f"{dataset_root}/images/train",
        labels_dir=f"{dataset_root}/labels/train",
        transform=train_transform,
        image_size=224
    )
    val_dataset = CustomDataset(
        images_dir=f"{dataset_root}/images/val",
        labels_dir=f"{dataset_root}/labels/val",
        transform=None,
        image_size=224
    )
    test_dataset = CustomDataset(
        images_dir=f"{dataset_root}/images/test",
        labels_dir=f"{dataset_root}/labels/test",
        transform=None,
        image_size=224
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=multibox_collate)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=multibox_collate)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=multibox_collate)

    # Training loop
    num_epochs = 30
    best_val_loss = float("inf")
    best_model_path = "model/best.pth"
    for epoch in range(num_epochs):
        train_loss = train_model(model, train_loader, optimizer, device, iou_module, reg_criterion, class_criterion, objectness_criterion, train_loss_logger, epoch=epoch+1)
        val_metrics = evaluate_model(model, val_loader, device, iou_module, threshold, threshold, epoch=epoch+1, 
                                    val_precision_logger=val_precision_logger, val_recall_logger=val_recall_logger, 
                                    val_miou_logger=val_miou_logger, val_class_accuracy_logger=val_class_accuracy_logger)
        scheduler.step()

        # Weighted loss combining all val metrics for selecting the best performing model
        val_loss = 2 * (1 - val_metrics["mean_iou"]) + (1 - val_metrics["class_accuracy"]) + (1 - val_metrics["precision"]) + (1 - val_metrics["recall"])

        # Save best model if the validation loss is better
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}")
        print(f"Validation Metrics: Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, mIoU: {val_metrics['mean_iou']:.4f}, Class Accuracy: {val_metrics['class_accuracy']:.4f}")

    # Load the best model for testing
    model.load_state_dict(torch.load(best_model_path))
    model.to(device)
    model.eval()

    # Run the best model on the test images and draw the outputs
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Draw bounding boxes for model predictions and export to output folder
    drawmodule = DrawModule()
    test_names = []
    test_folder = f"{dataset_root}/images/test"
    for filename in os.listdir(test_folder):
        if filename.lower().endswith((".jpg",".png")):
            test_names.append(filename)

    for name in test_names:
        test_path = os.path.join(test_folder, name)
        output_test_image(test_path, model, device, transform, drawmodule, idx=name.split(".")[0], threshold=threshold)

    # Evaluate the model on the test set
    test_metrics = evaluate_model(model, test_loader, device, iou_module, threshold, threshold)
    print(f"Test Metrics: Precision: {test_metrics['precision']:.4f}, Recall: {test_metrics['recall']:.4f}, mIoU: {test_metrics['mean_iou']:.4f}, Class Accuracy: {test_metrics['class_accuracy']:.4f}")

    # Plot training loss and validation metrics 
    plt.figure()
    plt.plot(train_loss_logger, label = "training losses")
    plt.legend()
    plt.title("test losses over epoch")

    plt.figure()
    plt.plot(val_precision_logger, label = "validation precision")
    plt.plot(val_recall_logger, label = "validation recall")
    plt.plot(val_miou_logger, label = "validation mIoU")
    plt.plot(val_class_accuracy_logger, label = "validation class accuracy")
    plt.legend()
    plt.title("Validation Metrics over Epoch")
    
    plt.show()

    return

if __name__ == "__main__":
    main()