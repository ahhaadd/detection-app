import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Load YOLO model
model = YOLO("best.pt")
classNames = ["armchair", "cabinet"]

st.title("Object Detection Demo")

# Webcam input
image = st.camera_input("Take a picture")

if image:
    # Convert image to OpenCV format
    image = np.array(Image.open(image))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Perform object detection
    results = model(image)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            confidence = box.conf[0].item()
            cls = int(box.cls[0].item())

            # Draw bounding boxes and labels
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 2)
            label = f"{classNames[cls]} {confidence:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    # Convert image back to PIL format and display
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image, caption="Processed Image", use_column_width=True)
