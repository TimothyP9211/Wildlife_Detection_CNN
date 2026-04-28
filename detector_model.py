import torch
import torch.nn as nn
from torch.nn import *

class DetectorModel(nn.Module):
    def __init__(self, baseModel, numClasses):
        super(DetectorModel, self).__init__
        
        self.baseModel = baseModel
        self.numClasses = numClasses
        self.baseModel.fc = Identity()
        
        # Regressor to output the bounding box coordinates
        self.regressor = Sequential(
            Linear(baseModel.fc.in_features, 128),
			ReLU(),
			Linear(128, 64),
			ReLU(),
			Linear(64, 32),
			ReLU(),
			Linear(32, 4),
			Sigmoid()
        )

        # Classifier to produce output label of image in bounding box, use dropout to combat overfitting
        self.classifier = Sequential(
			Linear(baseModel.fc.in_features, 512),
			ReLU(),
			Dropout(),
			Linear(512, 512),
			ReLU(),
			Dropout(),
			Linear(512, self.numClasses)
		)
    
    def forward(self, x):
        # Pass the inputs through the pretrained base model, then feed those outputs into the regression
        # and classification sectors to output the bbox coordinates and a label prediction
        features = self.baseModel(x)
        regression_bboxes = self.regressor(features)
        class_logits = self.classifier(features)
        return (regression_bboxes, class_logits)
