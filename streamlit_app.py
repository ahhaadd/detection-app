import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io

# Load YOLO model
model = YOLO("best.pt")
classNames = ["armchair", "cabinet"]

st.title("Real-Time Object Detection with YOLO")

# Function to process and annotate video frames
def process_frame(frame):
    # Convert frame to OpenCV format
    image = np.array(Image.open(io.BytesIO(frame)))
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

    return image

# Function to display video stream
def display_video():
    st.write("Webcam is streaming...")
    video_placeholder = st.empty()
    
    # Open video capture
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Failed to access webcam.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to read frame from webcam.")
            break

        # Process frame
        processed_frame = process_frame(frame)

        # Convert the processed frame to PIL format and display it
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(processed_frame, caption="Processed Frame", use_column_width=True)
        
        # Break the loop on user action
        if st.button("Stop Streaming"):
            break

    cap.release()

# Display video stream
display_video()
