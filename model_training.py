import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import nms
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

import numpy as np
import os

from tqdm import tqdm

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
def train_model(model, dataLoader, optimizer, device, iou_module, threshold,
          reg_criterion, class_criterion, objectness_criterion, train_loss_logger):
    model.train()
    running_loss = 0.0
    for images, target_bboxes, target_labels in dataLoader:
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
            IoU = iou_module(pred_bboxes[i], target_bboxes[i])
            matches = match_bboxes(IoU, threshold=threshold)

            for pred_idx, target_idx, iou_score in matches:
                # Object found in this box
                object_exists[i, pred_idx] = 1

                # Compute regression and classification loss for this matched box
                reg_loss += reg_criterion(pred_bboxes[i, pred_idx], target_bboxes[i, target_idx])
                class_loss += class_criterion(class_logits[i, pred_idx], target_labels[i, target_idx])
                match_count += 1

        # If no matches, then train only for objectness loss
        if match_count > 0:
            reg_loss /= match_count
            class_loss /= match_count
        else:
            reg_loss = torch.tensor(0.0, device=device)
            class_loss = torch.tensor(0.0, device=device)
        
        objectness_loss = objectness_criterion(objectness_logits, object_exists)

        loss = reg_loss + class_loss + objectness_loss

        # Backpropagation and optimization step
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        train_loss_logger.append(avg_loss.item())
    
    avg_loss = running_loss / len(dataLoader)
    return avg_loss

# Evaluate over one epoch
def evaluate_model(model, dataLoader, device, iou_module, threshold):
    model.eval()
    # TODO: Implement evaluation logic

def main():
    # TODO: Load dataset, train, evaluate, export model
    return

if __name__ == "__main__":
    main()