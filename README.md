# Xray Images Classifier for a covid dataset

X-ray chest images of a free covid dataset already split between covid, pneumonia and healthy patients

I got the dataset from: https://www.kaggle.com/pranavraikokte/covid19-image-dataset. It is pretty small with only 317.

The implementation is done in Pytorch and there is a possibility to choose between 2 pre-trained model: ResNet50 and VGG-16. Early stopping, Decay in learning rate factor, Data Augmentation also available.

These are the traning results i've got:

![alt text](https://github.com/chacoff/XrayImagesClassifier/blob/main/data/metrics.png?raw=true)

If is there anything you'd like to improve, i'll be happy to hearing from you.
