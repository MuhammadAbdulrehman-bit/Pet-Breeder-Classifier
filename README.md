# Pet Breed Classifier 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![Accuracy](https://img.shields.io/badge/Accuracy-90%25-brightgreen)]()

A deep learning model for classifying pet breeds, trained on the Oxford-IIIT Pet Dataset. Achieves 90% accuracy in identifying 37 different pet breeds.

![image](https://github.com/user-attachments/assets/6379e120-e922-4b6e-8b12-e8bcac1e5386)


## Features

- 🚀 High-accuracy classification (90% test accuracy)
- 🐶🐱 Supports 37 different pet breeds (12 cat breeds, 25 dog breeds)
- 📦 Easy-to-use prediction interface
- 🔧 Built with PyTorch/TensorFlow (choose one)
- 📊 Trained on Oxford-IIIT Pet Dataset

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/pet-breed-classifier.git
cd pet-breed-classifier
```

2. Install requirements:
 ```bash
   pip install streamlit tensorflow pillow numpy
   ```


### Prerequisites

- Python 3.10
- Tensorflow 2.8+



## Instructions

- The model has already been saved as .keras file. No need to train, just use the streamlit code
- Open the terminal of Visual studio code and run "streamlit run app.py".
- Or you can open your cmd, then navigate to the file path of the folder. Then type "streamlit run app.py"
- After doing that, a web application will open. Go to internet and select any images of cat or dog. Then save them and upload them on the web app. The model will predict it breed.
- **NOTE:** The model is limited in predicting only those breeds that are mentioned in this link: https://www.robots.ox.ac.uk/~vgg/data/pets/



## Dataset

This model was trained on the Oxford-IIIT Pet Dataset which contains:

- 7,349 images total
- 37 categories (12 cat breeds, 25 dog breeds)
- 50-200 images per breed
- Annotated with breed and segmentation masks



## Model Architecture

- Base Model: MobileNetV2 (pretrained on ImageNet)
- Custom Layers:
     - Data augmentation (random flip, rotation, zoom)
     - Global average pooling
     - Dense layer (128 units, ReLU activation)
     - Output layer (37 units, softmax activation)


### Training Details

- Input size: 224×224 RGB images
- Batch size: 32
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Epochs: 5
- Validation accuracy: ~90%



## How it works

1. User uploads an image through Streamlit interface
2. Image is resized to 224×224 and preprocessed
3. MobileNetV2 model makes predictions
4. Top 3 predictions with confidence scores are displayed



# License

This project is licensed under the MIT License - see the LICENSE file for details.



# Acknowledgments

- Oxford Visual Geometry Group for the pet dataset
- TensorFlow team for the deep learning framework
- Streamlit for the web application framework



# Examples:

![image](https://github.com/user-attachments/assets/630d13d9-8bc5-407c-9eb0-181474673787)

![image](https://github.com/user-attachments/assets/6ca75740-441b-43a5-b579-bb94964f218a)

![image](https://github.com/user-attachments/assets/6f62bfb5-e454-4606-bcb4-a5edc90e11b4)
