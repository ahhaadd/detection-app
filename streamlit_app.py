import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Load YOLO model
model = YOLO("best.pt")
classNames = ["armchair", "cabinet"]

# Streamlit app
st.title("Real-Time Object Detection")

# Placeholder for the video feed
image_placeholder = st.empty()

# Define a function to process the video stream
def process_frame(frame):
    results = model(frame)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            confidence = box.conf[0].item()
            cls = int(box.cls[0].item())

            # Draw bounding boxes and labels
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            label = f"{classNames[cls]} {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    return frame

# Capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        st.error("Failed to access webcam")
        break

    # Process frame
    frame = process_frame(frame)

    # Convert frame to RGB and display in Streamlit
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    image_placeholder.image(image, caption="Real-Time Object Detection", use_column_width=True)

# Release the capture
cap.release()
