import torch
import torch.nn as nn
from torch.nn import *

class DetectorModel(nn.Module):
    def __init__(self, baseModel, numClasses, numBBoxes):
        super(DetectorModel, self).__init__()
        
        self.baseModel = baseModel
        self.numClasses = numClasses
        self.numBBoxes = numBBoxes

        features = baseModel.fc.in_features
        self.baseModel.fc = Identity()
        
        # Regressor to output the bounding box coordinates
        self.regressor = Sequential(
            Linear(features, 128),
			ReLU(),
			Linear(128, 64),
			ReLU(),
			Linear(64, 32),
			ReLU(),
			Linear(32, 4 * numBBoxes),
			Sigmoid()
        )

        # Classifier to produce output label of image in bounding box, using dropout to combat overfitting
        self.classifier = Sequential(
			Linear(features, 512),
			ReLU(),
			Dropout(),
			Linear(512, 512),
			ReLU(),
			Dropout(),
			Linear(512, self.numClasses * numBBoxes)
		)

        # Objectness classifier to determine whether there is actually an object in the bounding box
        self.objectness_classifier = Sequential(
            Linear(features, 128),
            ReLU(),
            Linear(128, numBBoxes),
        )

    def forward(self, x):
        # Pass the inputs through the pretrained base model, then feed those outputs into the regression
        # and classification sectors to output fixed number of bbox coordinates and their label predictions
        features = self.baseModel(x)
        regression_bboxes = self.regressor(features).view(x.shape[0], self.numBBoxes, 4)
        class_logits = self.classifier(features).view(x.shape[0], self.numBBoxes, self.numClasses)
        objectness_logits = self.objectness_classifier(features).view(x.shape[0], self.numBBoxes, 1)
        return (regression_bboxes, class_logits, objectness_logits)
