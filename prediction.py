import tensorflow_hub as hub 
from tensorflow import keras
import pickle
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
import fitz
import cv2
import pytesseract
import numpy as np
from PIL import Image

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_data(data):
    cleaned_data = re.sub(r'[^a-zA-Z0-9\s.,]', '', data)
    cleaned_data = cleaned_data.lower()
    words = cleaned_data.split()
    words = [word for word in words if word not in stop_words]
    cleaned_data = ' '.join(words)
    return cleaned_data

def pdf_to_text(file:str):
    try:
        pdf_document = fitz.open(file)
        page = pdf_document.load_page(0)
        pix = page.get_pixmap(matrix=fitz.Matrix(1, 1))
        pil_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        image_cv2 = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        extracted_text = pytesseract.image_to_string(image_cv2)
        pdf_document.close()
        return extracted_text
    except Exception as e:
        return f"Error: {str(e)}"

def prediction(file):
    model = keras.models.load_model("model_&_encoder/form_classification_idfy.h5")
    text = pdf_to_text(file)
    text_c = [clean_data(text)]
    embedding_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    x_encoded = embedding_model(text_c).numpy()
    output = model.predict(x_encoded)
    loaded_encoder = pickle.load(open("model_&_encoder/label_encoder.pkl", 'rb'))
    output=loaded_encoder.inverse_transform(np.argmax(output, axis=1))
    return output
