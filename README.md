# Wildlife Detection CNN 
## Introduction
This personal project uses Pytorch to build and train a simple multi-object detection CNN to detect and classify seven classes of wildlife commonly found in North America. The main target use cases of such include detecting animals from still images (such as taken from a phone) as well as detecting animals from footage from wildlife cameras. The latter case can be useful in summarizing the different species of animal recorded in the wild for some given location which provides useful insight into the endemic life in that area.

The project includes a [Wildlife summary script](wildlife_summary.py) which takes an .mp4 recording from a wildlife cam and creates a summary of the frequency of the seven species that appear in the footage, outputting a frequency bar graph as well as a few (relatively) good shots of some of the animals. The detection CNN trained on thousands of images is utilized to make the detections and then filter by confidence.

## Model Construction
Object detection is more complicated than simple classfication as we have to not only train for classes, but also for the coordinates of the bounding boxes around the detected objects. Additionally, in the case of multiple objects, the model will have to be able to make a prediction for each object detected in the image which may vary the output dimension. To address this, the model will output a **fixed** number of bounding box predictions as well as an objectness value for each indicating how confident the model is that there is actually an object in the output bounding box. 

For simplicity, I will be using a pretrained model (in particular, ResNet50) for feature extraction and then will train three seperate models: a classifier for determining animal type, a regressor for bounding box predictions, and another classifer for objectness. The CNN pipeline is shown in figure 1 below.



## Data Collection



## Training

### Test Loss

### Validation Metrics


## Test Results



## Conclusions and Future Improvements
