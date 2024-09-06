import cv2
import numpy as np
from skimage.feature import hog
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from datetime import datetime
import os

image_path = r'C:\Z_My_Drive\CSE\Semester - 6\extra\FR'
attendance_file = r'C:\Z_My_Drive\CSE\Semester - 6\extra\FR\Attendance.csv'

IMAGE_SIZE = (128, 128)

def load_images_and_labels(path):
    images = []
    labels = []
    class_names = []
    label_encoder = LabelEncoder()
    
    for filename in os.listdir(path):
        img_path = os.path.join(path, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, IMAGE_SIZE)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            images.append(gray_img)
            label = os.path.splitext(filename)[0]
            if label not in class_names:
                class_names.append(label)
            labels.append(label)
    
    labels = label_encoder.fit_transform(labels)
    return images, labels, class_names

def extract_features(images):
    hog_features = []
    for img in images:
        features, _ = hog(img, block_norm='L2-Hys', visualize=True, pixels_per_cell=(16, 16), cells_per_block=(2, 2))
        hog_features.append(features)
    
    hog_features = np.array(hog_features)
    return hog_features

def train_classifier(features, labels):
    classifier = SVC(gamma='auto', kernel='linear')
    classifier.fit(features, labels)
    return classifier

def mark_attendance(name):
    with open(attendance_file, 'r+') as f:
        myDataList = f.readlines()
        nameList = [line.split(',')[0] for line in myDataList]
        if name not in nameList:
            now = datetime.now()
            time = now.strftime('%I:%M:%S:%p')
            date = now.strftime('%d-%B-%Y')
            f.write(f'{name}, {time}, {date}\n')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

images, labels, class_names = load_images_and_labels(image_path)
features = extract_features(images)
classifier = train_classifier(features, labels)

cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, IMAGE_SIZE)
        face_features = extract_features([face_resized])
        label = classifier.predict(face_features)[0]
        name = class_names[label].upper()
        color = (0, 255, 0)
        stroke = 2
        cv2.rectangle(img, (x, y), (x+w, y+h), color, stroke)
        cv2.rectangle(img, (x, y-35), (x+w, y), color, cv2.FILLED)
        cv2.putText(img, name, (x+7, y-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        mark_attendance(name)

    cv2.imshow('webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
