# Xray Images Classifier for a covid dataset

X-ray chest images of a free covid dataset already split between covid, pneumonia and healthy patients. The dataset is pretty small, only 317 images and are already sorted between training and validation.

**Conda**
```
conda activate <env>
conda install pip
pip freeze > requirements.txt
```
**PIP**
```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```
**git**
```
$ git clone https://github.com/chacoff/XrayImagesClassifier
```

**dataset**

download from: https://www.kaggle.com/pranavraikokte/covid19-image-dataset. 

**Implementation**

The implementation is done in Pytorch and there is a possibility to choose between 2 pre-trained model: ResNet50 and VGG-16. Early stopping, Decay in learning rate factor, Data Augmentation also available.

```
pre_model = 'resnet50'  # vgg16 or resnet50
```

**Training**
```
train.py
```
or
```
Train.ipynb
```


**Results**
These are the traning results i've got:

![alt text](https://github.com/chacoff/XrayImagesClassifier/blob/main/data/metrics.png?raw=true)

If is there anything you'd like to improve, i'll be happy to hearing from you.

<p align='center'>
<img src="https://github.com/chacoff/XrayImagesClassifier/blob/main/data/Covid_0.74_0100.jpeg" width="300">
<img src="https://github.com/chacoff/XrayImagesClassifier/blob/main/data/Pneumonia_0.95_0109.jpeg" width="300">
</p>
