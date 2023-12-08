# SKIN_CANCER_XAI
Welcome to our project on enhancing the credibility of skin cancer classification using Explainable Artificial Intelligence (XAI), specifically leveraging SHAP (SHapley Additive exPlanations). This repository hosts a comprehensive analysis and implementation showcasing how deep learning models, often viewed as 'black boxes', can be made interpretable and trustworthy in the critical domain of healthcare.

Skin cancer, a prevalent and potentially deadly disease, demands accurate and reliable diagnostic methods. While deep learning has significantly advanced automated skin lesion analysis, the inherent opaqueness of these models poses a challenge. Our project aims to bridge this gap by not only employing advanced neural networks for skin cancer classification but also incorporating SHAP to demystify the decision-making process of these models.

Through our detailed Exploratory Data Analysis (EDA) and the application of various convolutional neural network architectures, we delve deep into the predictive mechanisms. We further enhance the project's value by employing SHAP, a cutting-edge tool in explainable AI, to interpret the model predictions. This approach provides insights into what features the models are focusing on, thereby increasing the trustworthiness and credibility of the predictions made by these AI systems.

This project stands at the intersection of machine learning, healthcare, and ethics, representing our commitment to developing AI solutions that are not only powerful but also transparent and understandable.

Join us in exploring the fascinating world of explainable AI in skin cancer classification, where we strive to make AI decisions in healthcare more interpretable, credible, and trustworthy.

## Project Overview
This project aims to analyze skin lesions using deep learning models. The primary goal is to classify skin lesions into various categories, potentially aiding in early diagnosis of skin conditions, including malignant lesions. The project involves an Exploratory Data Analysis (EDA) followed by the application of various deep learning models.

## Files in the Repository
FINAL_Skin_EDA.ipynb: Notebook containing the exploratory data analysis of the skin lesion dataset.
FINAL_SKIN_vgg16-model.ipynb: Notebook implementing the VGG16 model for lesion classification.
FINAL_SKIN_resnet50-model.ipynb: Notebook implementing the ResNet50 model for lesion classification.
FINAL_SKIN_inceptionresnetv2-model.ipynb: Notebook implementing the InceptionResNetV2 model for lesion classification.
FINAL_SKIN_second-cnn-model.ipynb: Notebook implementing a custom CNN model for lesion classification.
FINAL_SHAP_ResNET.ipynb: Notebook for SHAP (SHapley Additive exPlanations) analysis on the ResNet model.

## How to Run the Project
1. Exploratory Data Analysis: Start with FINAL_Skin_EDA.ipynb to understand the dataset. This notebook provides insights into the distribution of different types of skin lesions and other relevant features in the dataset.
Model Training and Evaluation:
After completing the EDA, proceed to train various models. Each model is contained in its separate notebook:
2. FINAL_SKIN_vgg16-model.ipynb
3. FINAL_SKIN_resnet50-model.ipynb
4. FINAL_SKIN_inceptionresnetv2-model.ipynb
5. FINAL_SKIN_second-cnn-model.ipynb
Run each notebook to train the respective model on the dataset. Evaluate the models based on their performance metrics.
6. SHAP Analysis:
Once the models are trained and evaluated, use FINAL_SHAP_ResNET.ipynb to perform SHAP analysis on the ResNet model. This will help in understanding the model's decision-making process.

## Requirements
* Python 3.x
* TensorFlow
* Keras
* SHAP
* Matplotlib
* Seaborn
* Pandas
* Numpy

## Contributors
* Bharati Panigrahi
* Khushi Naik
* Tanaya Tamhankar
* Chetan Sah
