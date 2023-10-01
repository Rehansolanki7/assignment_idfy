# Text Classification Project README

This repository contains a text classification project that involves data preparation, model training, and prediction using machine learning and deep learning techniques. The project includes the following components:

1. `data_extraction.py`: A script for preparing and downloading data from the U.S. Securities and Exchange Commission (SEC) website, extracting images from PDFs, and performing OCR to extract text data.

2. `training.py`: A script for training a text classification model using the preprocessed text data, encoding the text using a sentence transformer model, and saving the trained model for future use.

3. `prediction.py`: A script for making predictions using the trained model, taking a PDF file as input, extracting text from it, and returning the predicted class label.

4. `main.py`: A FastAPI application that serves as an API endpoint for making text classification predictions. It integrates the prediction logic and allows users to make POST requests to get predictions for PDF files.

## Prerequisites
Before running any of the scripts or the FastAPI application, ensure you have the following dependencies installed:
- Python 3.10.12
- Installing required Python packages (install using `pip install -r requirements.txt`):

## Usage
1. Clone this repository to your local machine:

2. For `data_extraction.py`:
    - Modify the `url` variable in the script to specify the SEC website URL from which you want to fetch data.
    - Run the script using `python data_extraction.py`. It will download PDFs, extract images, and perform OCR,
     to prepare the data.

3. For `training.py`:
    - Organize your PDF images into class folders in a directory.
    - Modify the `pdf_directory` variable in the script to specify the path to the directory containing PDF images.
    - Run the script using `python training.py`. It will train a text classification model and save it for future use.

4. For `prediction.py`:
    - Ensure you have a pre-trained text classification model and label encoder saved in the "model_&_encoder" directory.
    - Modify the `file` variable in the script to specify the path to the PDF file you want to predict.
    - Run the script using `python prediction.py`. It will make predictions and return the class label.

5. For `main.py`:
    - Ensure you have the `prediction.py` script and a pre-trained model in the "model_&_encoder" directory.
    - Run the FastAPI application using `uvicorn main:app --reload`. It will serve as an API endpoint for text classification predictions.

## Customization
You can customize the scripts by modifying variables in the code to fit your specific use case.
You can further fine-tune the models or choose different pre-trained models as needed.

## Notes
- Make sure you have the required pre-trained models, label encoders, and dependencies installed.
- Organize your PDF images into class folders for supervised text classification.
- Need to install Tesseract OCR separately on your system if using Windows.
    To install Tesseract OCR on windows you can use the following the steps in the link:- https://linuxhint.com/install-tesseract-windows/

## Additional 
- Provided screenshot for architecture diagram, accuracy(training accuracy and validation accuracy) & confusion matric.
- Provided training data as well which have been extracted from the pdf and cleaned.