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
st.title("Real-Time Object Detection with YOLO")

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

# Capture video from webcam
cap = cv2.VideoCapture(0)

# Streamlit container for the video feed
stframe = st.empty()

# Run the video stream
while True:
    success, frame = cap.read()
    if not success:
        st.error("Failed to capture image from webcam.")
        break

    # Detect objects in the frame
    frame = detect_objects(frame)

    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Display the frame
    stframe.image(frame_rgb, channels='RGB', use_column_width=True)

    # Break loop on stop button click
    if st.button('Stop'):
        break

cap.release()
cv2.destroyAllWindows()
