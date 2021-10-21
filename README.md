# LUNG_DISEASE_DETECTION

# Introduction

Predicting whether the given image of the chest contains diseased or normal lungs using the VGG16 state of the art model.
We have to provide the x-ray image of the chest and the model is capable of  detecting.

# Description

Based on State of the Art convolutional neural network model VGG16 (16 layers: 13 convolutional layers and 3 dense layers one of them i.e., the output layer customized) using transfoer learning.

# Dataset

Availabel on kaggle at: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
Advised to fetch it and not download it separately. More on it ahead.

# Requirements

**General:**

Python>=3.6

**For Model Package:**

numpy>=1.19.0,<1.21.0
pandas>=1.2.0,<1.3.0
pydantic>=1.8.1,<1.9.0
scikit-learn>=0.24.0,<0.25.0
strictyaml>=1.3.2,<1.4.0
ruamel.yaml==0.16.12
feature-engine>=1.0.2,<1.1.0
joblib>=1.0.1,<1.1.0
kaggle==1.5.2
tensorflow==2.6.0
tensorflow-estimator==2.6.0
keras==2.6.0
opencv-python==4.5.3.56
opencv-contrib-python==4.5.3.56

**For API:**

uvicorn>=0.11.3,<0.12.0
fastapi>=0.64.0,<1.0.0
python-multipart>=0.0.5,<0.1.0
typing_extensions>=3.7.4,<3.8.0
loguru>=0.5.3,<0.6.0
python-json-logger>=0.1.11,<0.2.0
jinja2==3.0.2
python-multipart==0.0.5

**For Testing and Tooling of the Project:**

pytest>=6.2.3,<6.3.0
requests>=2.23.0,<2.24.0
black==20.8b1
flake8>=3.9.0,<3.10.0
mypy==0.812
isort==5.8.0

# Setup (Go to the suitable command prompt)

1] Clone the repository on the local system. Here the commands to be adjusted according to the path of the folder. Here the project is on my Desktop (Microsoft C-Drive)

2] For training the model

cd C:\Users\kshir\OneDrive\Desktop\DS_PROJECT_1\ds_project
tox -e fetch_data (fetching the data) </br>
tox -e train_test_package </br>

3] For installing the package locally

tox -e train_test_package </br> (use this to ensure that you have a trained model) </br>
cd C:\Users\kshir\OneDrive\Desktop\DS_PROJECT_1\ds_project\package
python setup.py sdist bdist-wheel
pip install -e .

4] For running the Api

cd C:\Users\kshir\OneDrive\Desktop\DS_PROJECT_1\ds_project
tox -e test_api
tox -e run

5] Once the application starts running go to http://localhost:8001/imageclassificationform </br>
   Store the live data in C:\Users\kshir\OneDrive\Desktop\DS_PROJECT_1\ds_project\live_data for uploading. </br>
    
# Contributors

VEDANG KSHIRSAGAR (vedang-04)



