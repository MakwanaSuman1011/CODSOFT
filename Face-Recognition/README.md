# Face Detection & Recognition (Flask, LBPH)

A Flask-based AI application to register faces and perform real-time face detection & recognition with confidence percentage. Detection runs only when you press the Start Detection button.

##Features

1) Face Registration

- Capture 5 images of a person via webcam.

- Automatically saves images to known_faces/<person_name>/.

- Trains LBPH recognizer after registration.

2) Face Detection & Recognition

- Detects faces in real-time using your webcam.

- Shows name and confidence percentage of recognized faces.

- Detection starts only after pressing a button.

3) Web Interface

- Simple, dark-themed UI.

- Live video feed displayed in the browser.


## Project structure

project/
│
├─ app.py                  
├─ train_recognizer.py     
├─ known_faces/            
├─ templates/
│   ├─ index.html          
│   └─ register.html       
└─ static/
    └─ style.css          


## Installation

1) Clone the project or download the files.

2) Create a virtual environment (optional but recommended)

python -m venv venv
venv\Scripts\activate    # Windows
source venv/bin/activate # Linux/Mac

3) Install required packages

pip install flask opencv-python opencv-contrib-python numpy

## How to Run

1) Start the Flask app

python app.py


2) Open your browser and go to:

http://127.0.0.1:5000/


3) Register a new face

- Enter your name and press Start Capture.

- Press c in the webcam window to capture 5 images.

- ESC cancels registration.

4) After registration, you are redirected to the detection page.

5) Press the Start Detection button to begin face recognition.