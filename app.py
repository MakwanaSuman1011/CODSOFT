from flask import Flask, render_template, request, redirect, url_for, Response
import cv2, os, json
import numpy as np

app = Flask(__name__)

DATA_DIR = "known_faces"
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

recognizer = None
label_map = {}
if os.path.exists("lbph_model.yml") and os.path.exists("label_map.json"):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("lbph_model.yml")
    with open("label_map.json", "r") as f:
        label_map = json.load(f)


@app.route('/')
def home():
    return redirect(url_for('register'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name'].strip()
        if not name:
            return "Enter a valid name."

        person_dir = os.path.join(DATA_DIR, name)
        os.makedirs(person_dir, exist_ok=True)

        cap = cv2.VideoCapture(0)
        count = 0
        print("Press 'c' to capture face images...")

        while count < 5:
            ret, frame = cap.read()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.imshow("Capture Face - Press 'c'", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('c') and len(faces) > 0:
                x, y, w, h = faces[0]
                face_img = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face_img, (200, 200))
                cv2.imwrite(os.path.join(person_dir, f"{count}.jpg"), face_resized)
                count += 1
                print(f"Captured image {count}/5")
            elif key == 27:  
                break

        cap.release()
        cv2.destroyAllWindows()

        from train_recognizer import train_and_save
        train_and_save()

        return redirect(url_for('index'))

    return render_template('register.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def gen_frames():
        cap = cv2.VideoCapture(0)
        while True:
            success, frame = cap.read()
            if not success:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                try:
                    face_resized = cv2.resize(face_roi, (200, 200))
                except Exception:
                    continue

                name = "Unknown"
                conf_percent = 0
                if recognizer is not None and len(label_map) > 0:
                    try:
                        label, conf = recognizer.predict(face_resized)
                        conf_percent = max(0, min(100, int(100 - conf)))  # % confidence
                        if conf < 100:  # threshold
                            name = label_map.get(str(label), "Unknown")
                        else:
                            name = "Unknown"
                    except Exception:
                        pass

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} ({conf_percent}%)", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        cap.release()

    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
