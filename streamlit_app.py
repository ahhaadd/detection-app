import streamlit as st
import cv2
from PIL import Image
import numpy as np
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("best.pt")

# Object classes
classNames = ["armchair", "cabinet"]
CONFIDENCE_THRESHOLD = 0.75

# Streamlit interface
st.title("Object Detection with YOLO")

# Function to detect objects and return the processed frame
def detect_objects(frame):
    results = model(frame, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            confidence = box.conf[0].item()
            if confidence >= CONFIDENCE_THRESHOLD:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cls = int(box.cls[0].item())
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2
                cv2.putText(frame, f"{classNames[cls]} {confidence:.2f}", org, font, fontScale, color, thickness)
    return frame

# Upload image or video file
uploaded_file = st.file_uploader("Choose an image or video file", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file is not None:
    # Load the file into a numpy array
    if uploaded_file.type.startswith('image'):
        image = Image.open(uploaded_file)
        frame = np.array(image)
    elif uploaded_file.type.startswith('video'):
        # For videos, read the first frame
        cap = cv2.VideoCapture(uploaded_file.read())
        ret, frame = cap.read()
        cap.release()
        if not ret:
            st.error("Failed to read video.")
            st.stop()
    
    # Detect objects in the frame
    frame = detect_objects(frame)

    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Display the frame
    st.image(frame_rgb, channels='RGB', use_column_width=True)
