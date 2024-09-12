import streamlit as st

st.title("Object Detection")
import streamlit as st
import cv2
import numpy as np
import math
from PIL import Image
from ultralytics import YOLO 

# Initialize YOLO model
model = YOLO("best.pt")

# Object classes
classNames = ["armchair", "cabinet"]

st.title("Real-Time Object Detection with YOLO")

# Capture webcam input
stframe = st.empty()
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        st.write("Error accessing webcam")
        break

    # Perform object detection
    results = model(img, stream=True)

    # Draw bounding boxes and labels
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # Draw class name and confidence
            confidence = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            label = f"{classNames[cls]} {confidence}"
            cv2.putText(img, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Convert image to RGB and display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    stframe.image(img_rgb, channels="RGB", use_column_width=True)

    # Stop the webcam capture loop if 'q' is pressed
    if st.button('Stop'):
        break

cap.release()
cv2.destroyAllWindows()
