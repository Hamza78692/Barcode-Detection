import streamlit as st
import cv2 as cv
import numpy as np
from PIL import Image

# Load Haar cascade classifiers
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
smile_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_smile.xml')

def detect_and_display(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.5, 5)
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv.putText(frame, 'Face', (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Detect eyes
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 3)
            cv.putText(roi_color, 'Eye', (ex, ey - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Detect smile
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.3, 5)
        for (sx, sy, sw, sh) in smiles:
            cv.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (255, 0, 0), 3)
            cv.putText(roi_color, 'Smile', (sx, sy - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return frame

def main():
    st.title("Real-time Face, Eye, and Smile Detection")
    
    # Placeholder for the video
    video_placeholder = st.empty()
    
    cap = cv.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image.")
            break
        
        frame = detect_and_display(frame)
        
        # Convert frame to RGB (OpenCV uses BGR by default)
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        
        video_placeholder.image(img, use_column_width=True)
        
        if st.button('Stop'):
            break
    
    cap.release()

if __name__ == "__main__":
    main()
