import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
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

healthy_images = load_images_from_folder('healthy/')
diseased_images = load_images_from_folder('diseased/')

healthy_features = preprocess_images(healthy_images)
diseased_features = preprocess_images(diseased_images)

X = np.vstack((healthy_features, diseased_features))
y = np.array([0] * len(healthy_features) + [1] * len(diseased_features)) 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = svm.SVC(kernel='linear')
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


joblib.dump(model, 'svm_model.pkl')
