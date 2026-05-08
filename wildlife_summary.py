import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image as PILImage
from collections import Counter

from torchvision.models import resnet50, ResNet50_Weights
from detector_model import DetectorModel
import matplotlib.pyplot as plt

import os
from tqdm import tqdm

# Parameters (matching model_training.py)
model_path = "model/best.pth"
video_path = "video/wildlife_cam1.mp4"
output_dir = "highlights/imgs/"
output_bgraph_dir = "highlights/"
num_classes = 7
num_bboxes = 8
confidence_threshold = 0.45

# Only look at one out of polling_rate frames
polling_rate = 10

animal_classes = {
    0: "Deer",
    1: "Moose",
    2: "Bear",
    3: "Fox",
    4: "Wolf",
    5: "Bison",
    6: "Squirrel",
}

# Loads the saved state of the model (with best weights after training) from the saved .pth file
def load_model(path, device, num_classes, num_bboxes):
    resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
    model = DetectorModel(baseModel=resnet, numClasses=num_classes, numBBoxes=num_bboxes)
    checkpoint = torch.load(path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model

# Check a single frame from the video and return detected labels and scores
def check_frame(model, device, frame, transform, confidence_threshold):
    # OpenCV uses BGR by default, convert to RGB 
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Make a prediction on the input frame
    input_tensor = transform(PILImage.fromarray(frame)).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_bboxes, class_logits, objectness_logits = model(input_tensor)

    # Filter the predictions by confidence threshold                 
    class_logits = class_logits[0]               
    objectness_logits = objectness_logits[0]
    pred_bboxes = pred_bboxes[0]  

    objectness_scores = torch.sigmoid(objectness_logits).squeeze(-1)
    class_probs = torch.softmax(class_logits, dim=-1)
    class_scores, class_labels = torch.max(class_probs, dim=-1)
    final_scores = class_scores * objectness_scores
    kept_indices = final_scores > confidence_threshold

    final_labels = class_labels[kept_indices]
    final_scores = final_scores[kept_indices]
    final_bboxes = pred_bboxes[kept_indices]

    # Bounding box size calculations (normalized)
    box_widths_norm = final_bboxes[:, 2]
    box_heights_norm = final_bboxes[:, 3]
    box_areas_norm = box_widths_norm * box_heights_norm

    return (final_labels.cpu(), final_scores.cpu(), box_areas_norm.cpu())

def main():
    print(f"cuda available: {torch.cuda.is_available()}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = load_model(path=model_path, device=device, num_classes=num_classes, num_bboxes=num_bboxes)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise RuntimeError(f"Failed to open video on path: {video_path}")
    
    prediction_count = Counter()
    best_images = {}

    # Only images with score = class_scores above this are considered for highlights
    min_score = 0.9

    # Only images with normalized size of bboxes above this are considered 
    min_size = 0.1

    frames_checked = 0
    total_frames = 0

    # Iterate through all frames in the video
    # Use tqdm for progress indication since video processing can take a while 
    for frame_idx in tqdm(range(int(video.get(cv2.CAP_PROP_FRAME_COUNT))), desc="Processing video"):
        ret, frame = video.read()
        if not ret:
            break
    
        # Check every once per polling_rate frames
        if total_frames % polling_rate == 0:
            labels, scores, sizes = check_frame(
                model=model,
                frame=frame,
                device=device,
                transform=transform,
                confidence_threshold=confidence_threshold
            )

            for label, score, size in zip(labels, scores, sizes):
                label_num = int(label.item())
                score_num = float(score.item())
                prediction_count[label_num] += 1
                
                # Check if the score of the predicted class is the current highest
                if ((label_num not in best_images or score_num > best_images[label_num]["score"]) 
                    and score_num > min_score and size > min_size):
                    best_images[label_num] = {
                        "score": score_num,
                        "frame_idx": frame_idx,
                        "frame": frame.copy()
                    }

            frames_checked += 1

        total_frames += 1

    video.release()

    for class_id, data in best_images.items():
        class_name = animal_classes.get(class_id, f"class_{class_id}")
        
        # Output the best frame per class 
        output_path = os.path.join(output_dir,f"{class_name}_score_{data['score']:.3f}.jpg")
        cv2.imwrite(output_path, data["frame"])

    # Match the class id to the animal names
    print("Frames with Counted Animals:")
    for class_id, count in prediction_count.items():
        class_name = animal_classes.get(class_id, f"class {class_id}")
        print(f"{class_name}: {count}")

    # Export a bar graph containing animal frequency of appearances
    class_ids = sorted(prediction_count.keys())
    labels = [animal_classes.get(class_id, f"class_{class_id}") for class_id in class_ids]
    counts = [prediction_count[class_id] for class_id in class_ids]

    total_count = sum(counts)
    if total_count > 0:
        normalized_counts = [count / total_count for count in counts]
    else:
        normalized_counts = [0 for _ in counts]

    plt.figure()
    plt.bar(labels, normalized_counts)
    plt.xlabel("Class")
    plt.ylabel("Frequency")
    plt.title(f"Wildlife frequency of appearance in {video_path}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_bgraph_dir}/wildlife_frequency.png")
    plt.close()

    return prediction_count

if __name__ == "__main__":
    main()