import os
import cv2
import numpy as np
import joblib

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

def preprocess_images(images, size=(64, 64)):
    processed_images = []
    for img in images:
        img = cv2.resize(img, size)
        img = img.flatten() 
        processed_images.append(img)
    return np.array(processed_images)

model = joblib.load('svm_model.pkl')

new_images = load_images_from_folder('new_images/') 

new_features = preprocess_images(new_images)

predictions = model.predict(new_features)

for i, prediction in enumerate(predictions):
    label = 'Healthy' if prediction == 0 else 'Diseased'
    print(f'Image {i + 1}: {label}')
