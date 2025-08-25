import os
import cv2
import numpy as np
import json

DATA_DIR = "known_faces"
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

def collect_training_data():
    X = []         
    y = []         
    label_map = {} 
    current_label = 0

    if not os.path.exists(DATA_DIR):
        print(f"Create '{DATA_DIR}/<person_name>/' and add face images for each person.")
        return X, y, label_map

    for person in sorted(os.listdir(DATA_DIR)):
        person_dir = os.path.join(DATA_DIR, person)
        if not os.path.isdir(person_dir):
            continue

        label_map[str(current_label)] = person

        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)
            if len(faces) == 0:
                continue

            x1, y1, w, h = faces[0]  
            face = cv2.resize(img[y1:y1+h, x1:x1+w], (200, 200))
            X.append(face)
            y.append(current_label)

        current_label += 1

    return X, y, label_map

def train_and_save():
    X, y, label_map = collect_training_data()
    if len(X) == 0:
        print("No training data found. Add images to known_faces/<name>/")
        return

    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
    except Exception as e:
        print("cv2.face not available. Make sure opencv-contrib-python is installed. Error:", e)
        return

    recognizer.train(X, np.array(y))
    recognizer.write("lbph_model.yml")

    # Save label map
    with open("label_map.json", "w") as f:
        json.dump(label_map, f)

    print("Trained and saved lbph_model.yml and label_map.json")

if __name__ == "__main__":
    train_and_save()
