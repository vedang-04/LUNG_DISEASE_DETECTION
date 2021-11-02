# LUNG_DISEASE_DETECTION

# Introduction

Predicting whether the given image of the chest contains diseased or normal lungs using the VGG16 state of the art model.
We have to provide the x-ray image of the chest and the model is capable of detecting whether the lungs are diseased or not.

# Description

Based on State of the Art convolutional neural network model VGG16 (16 layers: 13 convolutional layers and 3 dense layers one of them i.e., the output layer customized) using transfer learning.

# Dataset

Availabel on kaggle at: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia </br>
Advised to fetch it and not download it separately. More on it ahead. </br>
Kaggle Account required for fetching the dataset.</br>

# Requirements

**General:**

Python>=3.6 </br>

**For Model Package:**

numpy>=1.19.0,<1.21.0</br>
pandas>=1.2.0,<1.3.0</br>
pydantic>=1.8.1,<1.9.0</br>
scikit-learn>=0.24.0,<0.25.0</br>
strictyaml>=1.3.2,<1.4.0</br>
ruamel.yaml==0.16.12</br>
feature-engine>=1.0.2,<1.1.0</br>
joblib>=1.0.1,<1.1.0</br>
kaggle==1.5.2</br>
tensorflow==2.6.0</br>
tensorflow-estimator==2.6.0</br>
keras==2.6.0</br>
opencv-python==4.5.3.56</br>
opencv-contrib-python==4.5.3.56</br>

**For API:**</br>

uvicorn>=0.11.3,<0.12.0</br>
fastapi>=0.64.0,<1.0.0</br>
python-multipart>=0.0.5,<0.1.0</br>
typing_extensions>=3.7.4,<3.8.0</br>
loguru>=0.5.3,<0.6.0</br>
python-json-logger>=0.1.11,<0.2.0</br>
jinja2==3.0.2</br>
python-multipart==0.0.5</br>

**For Testing and Tooling of the Project:**</br>

pytest>=6.2.3,<6.3.0</br>
requests>=2.23.0,<2.24.0</br>
black==20.8b1</br>
flake8>=3.9.0,<3.10.0</br>
mypy==0.812</br>
isort==5.8.0</br>

# Setup (Go to the suitable command prompt)

1] Clone the repository on the local system. Here the commands to be adjusted according to the path of the folder. Here the project is on my Desktop (Microsoft C-Drive)</br>

2] For training the model

cd C:\Users\kshir\OneDrive\Desktop\LUNG_DISEASE_DETECTION\ds_project</br>
tox -e fetch_data (fetching the data) </br>
tox -e train_test_package </br>

3] For installing the package locally

tox -e train_test_package </br> (use this to ensure that you have a trained model) </br>
cd C:\Users\kshir\OneDrive\Desktop\LUNG_DISEASE_DETECTION\ds_project\package</br>
python setup.py sdist bdist-wheel</br>
pip install -e .</br>

4] For running the Api

cd C:\Users\kshir\OneDrive\Desktop\LUNG_DISEASE_DETECTION\ds_project</br>
tox -e test_api</br>
tox -e run</br>

5] Once the application starts running go to http://localhost:8001/imageclassificationform </br>
   Store the live data in C:\Users\kshir\OneDrive\Desktop\DS_PROJECT_1\ds_project\live_data for uploading. </br>
   
**More on this repository in this file:** https://drive.google.com/file/d/124NlI2jXW7TvRTKcupcb1Wy4FNHhNcyZ/view?usp=sharing</br>

**Indepth Analysis and explanation of this repository in this file:** https://drive.google.com/file/d/1QY_cTHHLySQ10xkEpLQB-pS7Dl6Pnley/view?usp=sharing</br>

**The files in the above mentioned two links are available in documents of this repository**</br>
    
# Contributors

VEDANG KSHIRSAGAR (vedang-04)</br>



