# Xray Images Classifier for a covid dataset

X-ray chest images of a free covid dataset already split between covid, pneumonia and healthy patients. I got the dataset from: https://www.kaggle.com/pranavraikokte/covid19-image-dataset. It is pretty small with only 317.

<b>INSTALL</b>
</br>

<b>Conda</b>
<font size="-1">conda activate <env></br>
conda install pip</br>
pip freeze > requirements_ingfisica.txt</br></font>
  
<b>PIP</b>
<p style="font-size:9x">python3 -m venv env</br>
source env/bin/activate</br>
pip install -r requirements.txt</br></p>

</br> </br>
<b>Implementation</b></br>
The implementation is done in Pytorch and there is a possibility to choose between 2 pre-trained model: ResNet50 and VGG-16. Early stopping, Decay in learning rate factor, Data Augmentation also available.

<b>Training</b>
<p style="font-size:9x">train.py</br>
or</br>
Train.ipynb</p>

These are the traning results i've got:

![alt text](https://github.com/chacoff/XrayImagesClassifier/blob/main/data/metrics.png?raw=true)

If is there anything you'd like to improve, i'll be happy to hearing from you.

<p align='center'>
  
<img src="https://github.com/chacoff/XrayImagesClassifier/blob/main/data/Covid_0.74_0100.jpeg" width="280">
<img src="https://github.com/chacoff/XrayImagesClassifier/blob/main/data/Pneumonia_0.95_0109.jpeg" width="280">
</p>
