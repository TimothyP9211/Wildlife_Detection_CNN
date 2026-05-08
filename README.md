# Wildlife Detection CNN 
## Introduction
This personal project uses Pytorch to build and train a simple multi-object detection CNN to detect and classify seven classes of wildlife commonly found in North America. The main target use cases of such include detecting animals from still images (such as taken from a phone) as well as detecting animals from footage from wildlife cameras. The latter case can be useful in summarizing the different species of animal recorded in the wild for some given location which provides useful insight into the endemic life in that area.

The project includes a [Wildlife summary script](wildlife_summary.py) which takes an .mp4 recording from a wildlife cam and creates a summary of the frequency of the seven species that appear in the footage, outputting a frequency bar graph as well as a few (relatively) good shots of some of the animals. The detection CNN trained on thousands of images is utilized to make the detections and then filter by confidence.

## Model Construction
Object detection is more complicated than simple classfication as we have to not only train for classes, but also for the coordinates of the bounding boxes around the detected objects. Additionally, in the case of multiple objects, the model will have to be able to make a prediction for each object detected in the image which may vary the output dimension. To address this, the model will output a **fixed** number of bounding box predictions as well as an objectness value for each indicating how confident the model is that there is actually an object in the output bounding box. 

For simplicity, I will be using a pretrained model (in particular, ResNet50) for feature extraction and then will train three seperate models: a classifier for determining animal type, a regressor for bounding box predictions, and another classifer for objectness. The CNN pipeline is shown in figure 1 below. 

![Model Architecture](figures/modelArc.png)
*Figure 1: Model Architecture*

## Data Collection
Most of the images of which the model is trained on are open-source, specifically from Kaggle or free stock image sources online. The [dataset class](dataset.py) used for loading the datasets expects the labels in YOLO (.txt) format, for which most of the images already included YOLO labels. However to further improve the dataset, I ended up using the tool labelImg to manually draw bounding boxes for around 400 images. 
The dataset is varied in the types of images of each animal class, including both close and far shots with one or more animals in distinct evironments.

As mentioned previously, there are seven animal classes each of which has a varying number of images shown in the figure below.
| Animal Class | ID | Total Imgs | # Train Imgs | # Val Imgs | # Test Imgs |
|---|---:|---:|---:|---:|---:|
| Deer | 0 | 995 | 875 | 100 | 20 |
| Moose | 1 | 221 | 176 | 30 | 15 |
| Bear | 2 | 674 | 579 | 85 | 10 |
| Fox | 3 | 648 | 560 | 70 | 18 |
| Wolf | 4 | 316 | 278 | 30 | 8 |
| Bison | 5 | 364 | 317 | 37 | 10 |
| Squirrel | 6 | 639 | 540 | 80 | 19 |

*Figure 2: Animal classes with image counts*

The total number of images in this dataset is 3857.

The dataset is structured in YOLO format with seperate folders for images and labels which split into train/test/val subfolders. 

## Training
I trained the model using the following [training script](model_training.py) as well as CUDA on an RTX 4070 Super for faster training times which allowed me to train much more frequentely while adjusting hyperparameters. 

### Test Loss
The loss functions used for each of the three parts of the model pipeline are fairly standard:

Object classifer: Cross-Entropy Loss

Regressor: Smooth L1 Loss

Objectness classifier: Binary Cross-Entropy Loss with logits

The overall training loss is a weighted sum of these individual loss functions:

**Training Loss = regressor_weight * reg_loss + classifier_weight * class_loss + objectness_weight * objectness_loss**

### Validation Metrics
#### Mean IoU
The Intersection Over Union (IoU) is a popular metric for evalualting the performance of a detector model. It is simply the area of the intersection over of area of the union between predicted and target bounding boxes which a value between 0 and 1 (higher IoU = better prediction). For this specific project, IoU was used not only as an evaluation metric, but also for matching bounding boxes between target and model predictions. The **mean IoU** is the average over all the predicted bounding boxes that the model detects, indicating regressor performance. 
#### Precision and Recall
Precision is measure of the % of positive predictions which are actually positive in the image, taking a large penalty from false positives. Meanwhile Recall is the % of objects detected by the model out of the object that are actually in the image, taking a large penalty from false negatives. The harmonic mean of Precision and Recall is the F1 score which indicates a balance of both. 
#### Class Accuracy
Class accuracy is simply the overall accuracy of the predicted classes for each object, indicating object classifier performance.

Taken together, the overall validation loss given by a weighted sum of each of these metrics:

**Validation Loss =  c_1 * (1 - Mean IoU) + c_2 * (1 - Class Accuracy) + c_3 * (1 - Precision) + c_4 * (1 - Recall)**

### Additional Design Choices
* Trained for 30 epochs
* Used Adam as the optimizer with learning rate of 0.0001 with a dynamic learning rate scheduler
* Transformed and normalized the data to 224x224 which is used by ResNet50
* Applied Augmentation on the dataset to increase the dataset diversity, specifically ColorJitter and GaussianBlur

## Test Results
Here are some of the animals that the model successfully detected from the test set:


![bison](figures/test_output_BISON_0007.jpg)
![squirrel](figures/test_output_squirrel_8.jpg)

![bear](figures/test_output_Bear_0341.jpg)

![wolf](figures/test_output_COOL_WOLF_0005.jpg)
![deer](figures/test_output_Deer_60.jpg)

![moose](figures/test_output_EPIC_MOOSE_0004.jpg)

![fox](figures/test_output_Fox_0003.jpg)



## Conclusions and Future Improvements


## References
Dataset Sources:

Prelabeled:

https://www.kaggle.com/datasets/shivamchaudharys/wildlife-detetion-yolo-74-class

https://www.kaggle.com/datasets/giotamoraiti/animal-object-detection-dataset-10-classes/data

Manually labelled:

https://www.kaggle.com/datasets/navidre/alberta-wildlife-dataset

https://www.kaggle.com/datasets/elhaddadmohamed/final-datasets-v2

Wildlife Cam footage:

https://www.youtube.com/@CanadaWildlifeCams

