import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow_hub as hub
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import os
import pytesseract
import pandas as pd
from PIL import Image

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def extract_text_from_image(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text

def pdf_to_dataframe(pdf_directory):
    data = [
        [extract_text_from_image(os.path.join(pdf_directory, class_name, pdf_image)), class_name]
        for class_name in os.listdir(pdf_directory)
        for pdf_image in os.listdir(os.path.join(pdf_directory, class_name))
    ]
    df = pd.DataFrame(data, columns=['data', 'class'])
    return df

def clean_data(data):
    cleaned_data = re.sub(r'[^a-zA-Z0-9\s.,]', '', data)
    cleaned_data = cleaned_data.lower()
    words = cleaned_data.split()
    words = [word for word in words if word not in stop_words]
    cleaned_data = ' '.join(words)
    return cleaned_data

def model_structure(input_shape):
    model = keras.Sequential([
    keras.layers.Input(shape=(input_shape[0],)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(.2),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(8, activation='softmax')
    ])
    return model

def confusionmatrics_accuracy(model,x_test,y_test):
    predicted_value = model.predict(x_test)
    confusion_matrix = confusion_matrix(y_test,predicted_value)
    accuracy_score = accuracy_score(y_test,predicted_value)
    return confusion_matrix,accuracy_score

def training(path):
    df = pdf_to_dataframe(path)
    df['data'] = df['data'].apply(clean_data)
    x = df["data"].tolist()
    y = df["class"]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    embedding_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    x_encoded = embedding_model(x).numpy()
    x_train, x_test, y_train, y_test = train_test_split(x_encoded, y, test_size=0.2, random_state=42)
    input_shape = [x_train.shape[1]]
    model = model_structure(input_shape)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    epochs = 60
    batch_size = 64
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    confusion_matrics, accuracy = confusionmatrics_accuracy(model,x_test,y_test)
    model.save("form_classification_idfy.h5")
    return str(confusion_matrics),str(accuracy)